#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define THREAD_PER_BLOCK 256

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
__global__ void reduce_add_in_load_float4(float *d_input, float *d_output)
{
    __shared__ float shared[THREAD_PER_BLOCK];
    float *input_begin = d_input + blockDim.x * blockIdx.x * 4;
    float4 f4 = FETCH_FLOAT4(input_begin[threadIdx.x * 4]);
    shared[threadIdx.x] = f4.x + f4.y + f4.z + f4.w;
    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i /= 2)
    {
        if (threadIdx.x < i)
            shared[threadIdx.x] += shared[threadIdx.x + i];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        d_output[blockIdx.x] = shared[0];
}


bool check(float *out, float *res, int n)
{
    for (int i = 0; i < n; i++)
    {
        if (abs(out[i] - res[i]) > 0.005)
            return false;
    }
    return true;
}

int main()
{

    const int N = 32 * 1024 * 1024;
    float *input = (float *)malloc(N * sizeof(float));
    float *d_input;
    cudaMalloc((void **)&d_input, N * sizeof(float));

    int block_num = N / THREAD_PER_BLOCK / 4;
    float *output = (float *)malloc((block_num) * sizeof(float));
    float *d_output;
    cudaMalloc((void **)&d_output, (block_num) * sizeof(float));
    float *result = (float *)malloc((block_num) * sizeof(float));
    for (int i = 0; i < N; i++)
    {
        input[i] = 2.0 * (float)drand48() - 1.0;
    }
    // cpu calc
    for (int i = 0; i < block_num; i++)
    {
        float cur = 0;
        for (int j = 0; j < THREAD_PER_BLOCK*4; j++)
        {
            cur += input[i * THREAD_PER_BLOCK*4 + j];
        }
        result[i] = cur;
    }

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(block_num, 1);
    dim3 Block(THREAD_PER_BLOCK, 1);
     for (int i = 0; i < 10; i++)
        reduce_add_in_load_float4<<<Grid, Block>>>(d_input, d_output);
    cudaMemcpy(output, d_output, block_num * sizeof(float), cudaMemcpyDeviceToHost);

    if (check(output, result, block_num))
        printf("the ans is right\n");
    else
    {
        printf("the ans is wrong\n");
        for (int i = 0; i < block_num; i++)
        {
            printf("%lf ", output[i]);
        }
        printf("\n");
    }

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
