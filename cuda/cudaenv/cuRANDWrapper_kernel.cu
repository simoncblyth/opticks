#include "cuRANDWrapper_kernel.hh"
#include "LaunchSequence.hh"
#include "curand_kernel.h"
#include "stdio.h"
#include "assert.h"

#define CUDA_SAFE_CALL( call) do {                                         \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
    exit(EXIT_FAILURE);                                                      \
    } } while (0)




curandState* create_rng_wrapper(
    LaunchSequence* launchseq
)
{
    unsigned int items = launchseq->getItems(); 

    curandState* dev_rng_states; 

    CUDA_SAFE_CALL( cudaMalloc((void**)&dev_rng_states, items*sizeof(curandState))); 
    
    return dev_rng_states ;
}





__global__ void init_rng(int threads_per_launch, int thread_offset, curandState* rng_states, unsigned long long seed, unsigned long long offset)
{
   // chroma approach is to recycle rng_states for each kernel launch 
   // in the cohort being propagated, which means the size of each kernel
   // launch is limited by timeouts occuring in any of the kernel launches 
   // including the curand_init one
   //
   // instead of doing this try having a state for every photon
   // and offsetting into it : the advantage is that changes to 
   // the CUDA launch configuration should not have any impact 
   // on the random number streams being consumed by the simulation (?) 
   //
   // But all these rng streams are rather expensive though, so 
   // should compare performace with chroma piecewise approach.
   // Maybe are just paying the expense at initialization ?
   //   

   int id = blockIdx.x*blockDim.x + threadIdx.x;
   if (id >= threads_per_launch) return;

   curand_init(seed, id + thread_offset , offset, &rng_states[id]);  

   // not &rng_states[id+thread_offset] as rng_states is offset already in kernel call
}


void init_rng_wrapper(
    LaunchSequence* launchseq,
    void* dev_rng_states, 
    unsigned long long seed, 
    unsigned long long offset
)
{
    assert(launchseq);
    for(unsigned int i=0 ; i < launchseq->getNumLaunches() ; i++ )
    {
        const Launch& launch = launchseq->getLaunch(i) ;

        launch.Summary("init_rng_wrapper");
    
        curandState* dev_rng_states_launch = (curandState*)dev_rng_states + launch.thread_offset ; 

        init_rng<<<launch.blocks_per_launch, launch.threads_per_block>>>( launch.threads_per_launch, launch.thread_offset, dev_rng_states_launch, seed, offset );

        CUDA_SAFE_CALL( cudaDeviceSynchronize() );
    } 
}






__global__ void test_rng(int threads_per_launch, int thread_offset, curandState* rng_states, float *a)
{ 
   //

    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= threads_per_launch) return;

    // NB no id offsetting on rng_states or a, as the offsetting
    // was done once in the kernel call 
    // this means thread_offset argument not used

    curandState rng = rng_states[id];  
    
    a[id] = curand_uniform(&rng);

    rng_states[id] = rng;   
}



void test_rng_wrapper(
    LaunchSequence* launchseq,
    void* dev_rng_states,
    float* host_a
)
{
    assert(launchseq);
    //launchseq->Summary("test_rng_wrapper");

    unsigned int items = launchseq->getItems(); 

    float* dev_a; 
    CUDA_SAFE_CALL(cudaMalloc((void**)&dev_a, items*sizeof(float)));


    for(unsigned int i=0 ; i < launchseq->getNumLaunches() ; i++ )
    {
        const Launch& launch = launchseq->getLaunch(i) ;

        //launch.Summary("test_rng_wrapper");
    
        curandState* dev_rng_states_launch = (curandState*)dev_rng_states + launch.thread_offset ; 
        float*       host_a_launch = host_a + launch.thread_offset ; 
        float*       dev_a_launch = dev_a + launch.thread_offset ; 

        test_rng<<<launch.blocks_per_launch, launch.threads_per_block>>>( launch.threads_per_launch, launch.thread_offset, dev_rng_states_launch, dev_a_launch );

        CUDA_SAFE_CALL( cudaDeviceSynchronize() );

        CUDA_SAFE_CALL( cudaMemcpy(host_a_launch, dev_a_launch, launch.threads_per_launch*sizeof(float), cudaMemcpyDeviceToHost) ); 
    } 

    CUDA_SAFE_CALL( cudaFree(dev_a) );

    CUDA_SAFE_CALL( cudaDeviceSynchronize() );
}











