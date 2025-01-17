#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define THREAD_PER_BLOCK 256

__device__ void warpReduce(volatile float *cache, unsigned int tid)
{
    cache[tid] += cache[tid + 32];

    cache[tid] += cache[tid + 16];

    cache[tid] += cache[tid + 8];

    cache[tid] += cache[tid + 4];

    cache[tid] += cache[tid + 2];

    cache[tid] += cache[tid + 1];

}
template <unsigned int NUM_PER_BLOCK, unsigned int NUM_PER_THREAD>
__global__ void reduce_optimize_block(float *d_input, float *d_output)
{
    int tid = threadIdx.x;
    __shared__ float shared[THREAD_PER_BLOCK];

    float *input_begin = d_input + NUM_PER_BLOCK * blockIdx.x;
    shared[tid] = 0;
    for (int i = 0; i < NUM_PER_THREAD; i++)
        shared[tid] += input_begin[tid + i * THREAD_PER_BLOCK];
    __syncthreads();

    if (THREAD_PER_BLOCK >= 512)
    {
        if (tid < 256)
            shared[tid] += shared[tid + 256];
        __syncthreads();
    }
    if (THREAD_PER_BLOCK >= 256)
    {
        if (tid < 128)
            shared[tid] += shared[tid + 128];
        __syncthreads();
    }
    if (THREAD_PER_BLOCK >= 64)
    {
        if (tid < 64)
            shared[tid] += shared[tid + 64];
        __syncthreads();
    }

    if (tid < 32)
    {
        warpReduce(shared, tid);
    }
    if (tid == 0)
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

    constexpr int block_num = 1024;
    constexpr int num_per_block = N / block_num;
    constexpr int num_per_thread = num_per_block / THREAD_PER_BLOCK;
    float *output = (float *)malloc(block_num * sizeof(float));
    float *d_output;
    cudaMalloc((void **)&d_output, block_num * sizeof(float));
    float *result = (float *)malloc(block_num * sizeof(float));
    for (int i = 0; i < N; i++)
    {
        input[i] = 2.0 * (float)drand48() - 1.0;
    }
    // cpu calc
    for (int i = 0; i < block_num; i++)
    {
        float cur = 0;
        for (int j = 0; j < num_per_block; j++)
        {
            cur += input[i * num_per_block + j];
        }
        result[i] = cur;
    }

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(block_num, 1);
    dim3 Block(THREAD_PER_BLOCK, 1);
    for (int i = 0; i < 10; i++)
        reduce_optimize_block<num_per_block, num_per_thread><<<Grid, Block>>>(d_input, d_output);
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