
__global__ void CRng_generate(int threads_per_launch, curandState* rng_states, float* d_arr )
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= threads_per_launch) return;

    float u = curand_uniform(&rng_states[id]); 
    d_arr[id] = u ;   
}


