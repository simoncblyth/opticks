#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "qcurand.h"

__global__ void setup_curand_kernel(curandState_t* states, unsigned long seed, size_t num_values) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < num_values) {
        curand_init(seed, id, 0, &states[id]);
    }
}

template <typename T>
__global__ void generate_exponential_kernel(curandState_t* states, T* d_values, size_t num_values) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < num_values) {
        d_values[id] = qcurand_shoot_exponential<T>(&states[id]);
    }
}

template <typename T>
static void launch_generate_exponential(T* h_values, size_t num_values, unsigned long seed) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_values + threadsPerBlock - 1) / threadsPerBlock;

    curandState_t* d_states = nullptr;
    T* d_values = nullptr;

    cudaMalloc(&d_states, num_values * sizeof(curandState_t));
    cudaMalloc(&d_values, num_values * sizeof(T));

    setup_curand_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_states, seed, num_values);
    generate_exponential_kernel<T><<<blocksPerGrid, threadsPerBlock>>>(d_states, d_values, num_values);
    cudaDeviceSynchronize();

    cudaMemcpy(h_values, d_values, num_values * sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_values);
    cudaFree(d_states);
}

// Concrete C-interface symbols for GCC
extern "C" {

void run_qcurand_test_float(float* values, size_t num_values, unsigned long seed) {
    launch_generate_exponential<float>(values, num_values, seed);
}

void run_qcurand_test_double(double* values, size_t num_values, unsigned long seed) {
    launch_generate_exponential<double>(values, num_values, seed);
}

} // extern "C"
