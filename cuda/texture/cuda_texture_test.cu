/*

CUDA texture loading and use with tex1D 
========================================= 

Example from MisterAnderson42

* https://devtalk.nvidia.com/default/topic/381335/setting-up-for-tex1d-how-to-load-cuda-array-for-tex1d-/

::

    delta:texture blyth$ cuda-
    delta:texture blyth$ nvcc -o cuda_texture_test cuda_texture_test.cu 
    delta:texture blyth$ ./cuda_texture_test && rm cuda_texture_test
    0.000000
    0.000000
    0.500000
    1.000000
    1.500000
    2.000000
    2.500000
    ...
    124.500000
    125.000000
    125.500000
    126.000000
    126.500000
    127.000000

*/

#include <stdio.h>

#define CUDA_SAFE_CALL( call) do {                                         \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
    exit(EXIT_FAILURE);                                                      \
    } } while (0)

#define CUT_CHECK_ERROR(errorMessage) do {                                 \
    cudaThreadSynchronize();                                                \
    cudaError_t err = cudaGetLastError();                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    } } while (0)


texture<float, 1, cudaReadModeElementType> tex;

__global__ void kernel(int M, float *d_out)
{
    float v = float(threadIdx.x) / float(blockDim.x) * float(M);
    float x = tex1D(tex, v);
    //printf("%f\n", x); // for deviceemu testing
    d_out[threadIdx.x] = x;
}

int main()
{
    int N = 256;
    float *d_out;
    CUDA_SAFE_CALL( cudaMalloc((void**)&d_out, sizeof(float) * N) ); // memory for output

    int M = N/2;
    // make an array half the size of the output
    cudaArray* cuArray;
    CUDA_SAFE_CALL (cudaMallocArray (&cuArray, &tex.channelDesc, M, 1));
    CUDA_SAFE_CALL (cudaBindTextureToArray (tex, cuArray));
    tex.filterMode = cudaFilterModeLinear;

    // data fill array with increasing values
    float *data = (float*)malloc(M*sizeof(float));

    for (int i = 0; i < M; i++) data[i] = float(i);

    CUDA_SAFE_CALL( cudaMemcpyToArray(cuArray, 0, 0, data, sizeof(float)*M, cudaMemcpyHostToDevice) );

    kernel<<<1, N>>>(M, d_out);

    float *h_out = (float*)malloc(sizeof(float)*N);

    CUDA_SAFE_CALL( cudaMemcpy(h_out, d_out, sizeof(float)*N, cudaMemcpyDeviceToHost) );

    for (int i = 0; i < N; i++) printf("%f\n", h_out[i]);

    free(h_out);
    free(data);
    cudaFreeArray(cuArray);
    cudaFree(d_out);
}
