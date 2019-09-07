/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#include <cstdio>
#include <cassert>

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlanguage-extension-token"
#endif

#include "cuRANDWrapper_kernel.hh"
#include "LaunchSequence.hh"
#include "curand_kernel.h"

#ifdef __clang__
#pragma clang diagnostic pop
#endif





#define CUDA_SAFE_CALL( call) do {                                         \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
    exit(EXIT_FAILURE);                                                      \
    } } while (0)



/**
allocate_rng_wrapper
---------------------

Allocates curandState device buffer sized to hold the number
of items from LaunchSequence and returns pointer to it. 

**/

CUdeviceptr allocate_rng_wrapper( LaunchSequence* launchseq)
{
    unsigned int items = launchseq->getItems(); 
    size_t nbytes = items*sizeof(curandState) ;
    int value = 0 ; 

    CUdeviceptr dev_rng_states ;

    CUDA_SAFE_CALL( cudaMalloc((void**)&dev_rng_states, nbytes )); 

    CUDA_SAFE_CALL( cudaMemset((void*)dev_rng_states, value, nbytes ));
    
    return dev_rng_states ;
}


/**
free_rng_wrapper
-------------------

Frees device buffer provided as argument.

**/

void free_rng_wrapper( CUdeviceptr dev_rng_states )
{
    CUDA_SAFE_CALL( cudaFree((void*)dev_rng_states));
}


/**
copytohost_rng_wrapper
-----------------------

1. allocates host memory for item count from the launchseq 
2. cudaMemcpy from device buffer pointer provided in argument to the host buffer 
3. returns pointer to host buffer

**/

curandState* copytohost_rng_wrapper( LaunchSequence* launchseq, CUdeviceptr dev_rng_states)
{
    unsigned items = launchseq->getItems(); 

    void* host_rng_states = malloc(sizeof(curandState)*items);

    CUDA_SAFE_CALL( cudaMemcpy(host_rng_states, (void*)dev_rng_states, sizeof(curandState)*items, cudaMemcpyDeviceToHost) );

    CUDA_SAFE_CALL( cudaDeviceSynchronize() );

    return (curandState*)host_rng_states ;
}

/**
copytodevice_rng_wrapper
--------------------------

1. allocates device buffer sized for launchseq items
2. cudaMemcpy from host buffer provided in argument
3. returns device pointer

**/

CUdeviceptr copytodevice_rng_wrapper( LaunchSequence* launchseq, void* host_rng_states)
{

    unsigned int items = launchseq->getItems(); 

    CUdeviceptr dev_rng_states; 

    CUDA_SAFE_CALL( cudaMalloc((void**)&dev_rng_states, items*sizeof(curandState))); 

    CUDA_SAFE_CALL( cudaMemcpy((void*)dev_rng_states, host_rng_states, sizeof(curandState)*items, cudaMemcpyHostToDevice) );

    CUDA_SAFE_CALL( cudaDeviceSynchronize() );

    return dev_rng_states ;
}





/**
before_kernel
--------------

* create start and stop events and record the start

**/

void before_kernel( cudaEvent_t& start, cudaEvent_t& stop )
{
    CUDA_SAFE_CALL( cudaEventCreate( &start ) );
    CUDA_SAFE_CALL( cudaEventCreate( &stop ) );
    CUDA_SAFE_CALL( cudaEventRecord( start,0 ) );
}


/**
after_kernel
--------------

* record the stop, returning elapsed time in kernel time argument

**/

void after_kernel( cudaEvent_t& start, cudaEvent_t& stop, float& kernel_time )
{
    CUDA_SAFE_CALL( cudaEventRecord( stop,0 ) );
    CUDA_SAFE_CALL( cudaEventSynchronize(stop) );

    CUDA_SAFE_CALL( cudaEventElapsedTime(&kernel_time, start, stop) );
    CUDA_SAFE_CALL( cudaEventDestroy( start ) );
    CUDA_SAFE_CALL( cudaEventDestroy( stop ) );

    CUDA_SAFE_CALL( cudaDeviceSynchronize() );
}



void devicesync_wrapper()
{
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );
}



/**
init_rng
----------

Invokes curand_init with resulting curandState written into rng_states 
of the argument.

The thread_offset is a technicality from doing multiple
launches to complete the initialize.
As rng_state is already offset in the kernerl call, NOT doing
the below as its cleaner for id to be local to the launch::

   &rng_states[id+thread_offset] 


Chroma approach was to recycle rng_states for each kernel launch 
in the cohort being propagated, which means the size of each kernel
launch is limited by timeouts occuring in any of the kernel launches 
including the curand_init one
   
Instead of doing this try having a state for every photon
and offsetting into it : the advantage is that changes to 
the CUDA launch configuration should not have any impact 
on the random number streams being consumed by the simulation (?) 
  
But all these rng streams are rather expensive though, so 
should compare performace with chroma piecewise approach.
Maybe are just paying the expense at initialization ?
   
(On macOS) curand_init runs 10x slower for large thread_offset ? 
starting from 262144 running the kernel launch sequence in reverse 
confirms this finding 
   
* :google:`curand_init slow with large sequence numbers`





From cuda-curand CURAND_Library.pdf Chapter 3::

    __device__ void curand_init (
    unsigned long long seed, 
    unsigned long long sequence,
    unsigned long long offset, 
    curandState_t *state )

The curand_init() function sets up an initial state allocated by the caller using the
given seed, sequence number, and offset within the sequence. Different seeds are
guaranteed to produce different starting states and different sequences.
...
Sequences generated with different seeds usually do not have statistically correlated
values, but some choices of seeds may give statistically correlated sequences. Sequences
generated with the same seed and different sequence numbers will not have statistically
correlated values.

For the highest quality parallel pseudorandom number generation, each experiment
should be assigned a unique seed. Within an experiment, each thread of computation
should be assigned a unique sequence number. If an experiment spans multiple kernel
launches, it is recommended that threads between kernel launches be given the same
seed, and sequence numbers be assigned in a monotonically increasing way. If the same
configuration of threads is launched, random state can be preserved in global memory
between launches to avoid state setup time.


Opticks Approach
~~~~~~~~~~~~~~~~~~~~

Photon record_id used as the curand sequence number  
with seed and offset set as zero in cuRANDWrapperTest. 

**/

__global__ void init_rng(int threads_per_launch, int thread_offset, curandState* rng_states, unsigned long long seed, unsigned long long offset)
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= threads_per_launch) return;

    curand_init(seed, id + thread_offset , offset, &rng_states[id]);  
}


/**
init_rng_wrapper
-----------------

Loops over launchseq NumLaunches invoking init_rng
which writes curandStates into offset device buffer locations.

**/

void init_rng_wrapper( LaunchSequence* launchseq, CUdeviceptr dev_rng_states, unsigned long long seed, unsigned long long offset)
{
    cudaEvent_t start, stop ;

    for(unsigned i=0 ; i < launchseq->getNumLaunches() ; i++ )
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



/**
test_rng
-----------

Use the rng_states curandState of the argument, offset by id, 
to generate some random floats.

NB no id offsetting on rng_states or a, as the offsetting
was done once in the kernel call 
this means thread_offset argument not used

curandState struct contains double boxmuller_extra_double
that causes demoting to float warnings in the below.
Stanley Seibert judges it to be benign.

* http://lists.tiker.net/pipermail/pycuda/2011-December/003513.html   

**/

__global__ void test_rng(int threads_per_launch, int thread_offset, curandState* rng_states, float *a,  bool update_states )
{ 
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= threads_per_launch) return;

    curandState rng = rng_states[id];   // copy from global to register
    
    a[id] = curand_uniform(&rng);   

    if(update_states) rng_states[id] = rng;               // update from register to global
}



/**
test_rng_wrapper
----------------

1. allocate device buffer dev_a with room for one float per item 
2. loop over launchseq launches invoking test_rng launches 
   using the curandStates provided in the argument 
3. for each thread of the launch a single random float is 
   generated which is stored in dev_a
4. copy dev_a to host_a of argument and free on device


**/

void test_rng_wrapper(
    LaunchSequence* launchseq,
    CUdeviceptr dev_rng_states,
    float* host_a, 
    bool update_states
)
{
    cudaEvent_t start, stop ;

    unsigned int items = launchseq->getItems(); 

    float* dev_a; 
    CUDA_SAFE_CALL(cudaMalloc((void**)&dev_a, items*sizeof(float)));

    for(unsigned i=0 ; i < launchseq->getNumLaunches() ; i++ )
    {
        Launch& launch = launchseq->getLaunch(i) ;

        curandState* dev_rng_states_launch = (curandState*)dev_rng_states + launch.thread_offset ; 
        float*       dev_a_launch = dev_a + launch.thread_offset ; 

        before_kernel( start, stop );

        test_rng<<<launch.blocks_per_launch, launch.threads_per_block>>>( launch.threads_per_launch, launch.thread_offset, dev_rng_states_launch, dev_a_launch, update_states );

        after_kernel( start, stop, launch.kernel_time );
    } 

    CUDA_SAFE_CALL( cudaMemcpy(host_a, dev_a, items*sizeof(float), cudaMemcpyDeviceToHost) ); 

    CUDA_SAFE_CALL( cudaFree(dev_a) );

    CUDA_SAFE_CALL( cudaDeviceSynchronize() );
}



