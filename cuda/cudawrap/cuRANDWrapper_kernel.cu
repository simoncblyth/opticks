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
   //
   // curand_init runs 10x slower for large thread_offset ? starting from 262144
   // running the kernel launch sequence in reverse confirms this finding 
   //
   // :google:`curand_init slow with large sequence numbers`
   //
}



void before_kernel( cudaEvent_t& start, cudaEvent_t& stop )
{
    CUDA_SAFE_CALL( cudaEventCreate( &start ) );
    CUDA_SAFE_CALL( cudaEventCreate( &stop ) );
    CUDA_SAFE_CALL( cudaEventRecord( start,0 ) );
}

void after_kernel( cudaEvent_t& start, cudaEvent_t& stop, float& kernel_time )
{
    CUDA_SAFE_CALL( cudaEventRecord( stop,0 ) );
    CUDA_SAFE_CALL( cudaEventSynchronize(stop) );

    CUDA_SAFE_CALL( cudaEventElapsedTime(&kernel_time, start, stop) );
    CUDA_SAFE_CALL( cudaEventDestroy( start ) );
    CUDA_SAFE_CALL( cudaEventDestroy( stop ) );
}



void init_rng_wrapper(
    LaunchSequence* launchseq,
    void* dev_rng_states, 
    unsigned long long seed, 
    unsigned long long offset
)
{
    cudaEvent_t start, stop ;

    for(unsigned int i=0 ; i < launchseq->getNumLaunches() ; i++ )
    {
        Launch& launch = launchseq->getLaunch(i) ;
    
        curandState* dev_rng_states_launch = (curandState*)dev_rng_states + launch.thread_offset ; 

        before_kernel( start, stop );

        init_rng<<<launch.blocks_per_launch, launch.threads_per_block>>>( launch.threads_per_launch, launch.thread_offset, dev_rng_states_launch, seed, offset );

        after_kernel( start, stop, launch.kernel_time );

        launch.Summary("init_rng_wrapper");
    } 
    launchseq->Summary("init_rng_wrapper");
}






__global__ void test_rng(int threads_per_launch, int thread_offset, curandState* rng_states, float *a)
{ 
   //

    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= threads_per_launch) return;

    // NB no id offsetting on rng_states or a, as the offsetting
    // was done once in the kernel call 
    // this means thread_offset argument not used

    curandState rng = rng_states[id];   // copy from global to register
    
    a[id] = curand_uniform(&rng);   

    rng_states[id] = rng;            // update from register to global

    //
    // curandState struct contains
    //        double boxmuller_extra_double
    // that causes demoting to float warnings
    // for the two above lines. 
    // Stanley Seibert judges it to be benign,
    //
    //   http://lists.tiker.net/pipermail/pycuda/2011-December/003513.html   
    //   
}



void test_rng_wrapper(
    LaunchSequence* launchseq,
    void* dev_rng_states,
    float* host_a
)
{
    cudaEvent_t start, stop ;

    unsigned int items = launchseq->getItems(); 

    float* dev_a; 
    CUDA_SAFE_CALL(cudaMalloc((void**)&dev_a, items*sizeof(float)));

    for(unsigned int i=0 ; i < launchseq->getNumLaunches() ; i++ )
    {
        Launch& launch = launchseq->getLaunch(i) ;

        curandState* dev_rng_states_launch = (curandState*)dev_rng_states + launch.thread_offset ; 
        float*       dev_a_launch = dev_a + launch.thread_offset ; 

        before_kernel( start, stop );

        test_rng<<<launch.blocks_per_launch, launch.threads_per_block>>>( launch.threads_per_launch, launch.thread_offset, dev_rng_states_launch, dev_a_launch );

        after_kernel( start, stop, launch.kernel_time );
    } 

    CUDA_SAFE_CALL( cudaMemcpy(host_a, dev_a, items*sizeof(float), cudaMemcpyDeviceToHost) ); 

    CUDA_SAFE_CALL( cudaFree(dev_a) );

    CUDA_SAFE_CALL( cudaDeviceSynchronize() );
}











