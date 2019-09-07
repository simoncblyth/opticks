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


//  https://devtalk.nvidia.com/default/topic/498171/how-to-get-same-output-by-curand-in-cpu-and-gpu/

/*

For CURAND, we use the seed to pick a random spot to start, then cut the long
sequence into 4096 chunks each spaced 2^67 positions apart. The host API lets
you grab blocks of results from this shuffled sequence. If you request 8192
results, you will get the first result from each of the 4096 chunks, then the
second result from each of the 4096 chunks.

For the device API using curand_init(), you explicitly give the subsequence
number and manage the threads yourself. 

If you want to exactly match the results from the host API you need 
to launch 4096 total threads, then have each
one call curand_init() with the same seed and subsequence numbers from 0 to
4095. Then you need to store the results in a coalesced strided manner; thread
0 goes first with one value, then next in memory is thread 1 with one value,
then thread 2, etc.. 

The reason you are seeing the number 8192 is because you are generating double
precision values. Each double result uses 2 32-bit results.

Let me know if that doesn't explain things.


From cuRANDWrapper_kernel.cu::

    093 __global__ void init_rng(int threads_per_launch, int thread_offset, curandState* rng_states, unsigned long long seed, unsigned long long offset)
     94 {
     95    // chroma approach is to recycle rng_states for each kernel launch 
     96    // in the cohort being propagated, which means the size of each kernel
     97    // launch is limited by timeouts occuring in any of the kernel launches 
     98    // including the curand_init one
     99    //
    100    // instead of doing this try having a state for every photon
    101    // and offsetting into it : the advantage is that changes to 
    102    // the CUDA launch configuration should not have any impact 
    103    // on the random number streams being consumed by the simulation (?) 
    104    //
    105    // But all these rng streams are rather expensive though, so 
    106    // should compare performace with chroma piecewise approach.
    107    // Maybe are just paying the expense at initialization ?
    108    //   
    109 
    110    int id = blockIdx.x*blockDim.x + threadIdx.x;
    111    if (id >= threads_per_launch) return;
    112 
    113    curand_init(seed, id + thread_offset , offset, &rng_states[id]);
    114 
    115    // not &rng_states[id+thread_offset] as rng_states is offset already in kernel call
    116    //
    117    // curand_init runs 10x slower for large thread_offset ? starting from 262144
    118    // running the kernel launch sequence in reverse confirms this finding 
    119    //
    120    // :google:`curand_init slow with large sequence numbers`
    121    //
    122 }
    123 




*/

#include <stdio.h>
#include <curand_kernel.h>

__global__ void kernel()
{
    curandState rngState;
    curand_init(1234,0,0,&rngState);
    for(int i=0;i<10;i++)
    {
        //curand_init(1234,i,0,&rngState); // i: sequence number
        //printf("%lf ",curand_uniform_double(&rngState));
        printf("%f ",curand_uniform(&rngState));
    }
    printf("\n");
}

int main()
{
    int *foo; // for in-kernel printf
    cudaMalloc(&foo,sizeof(int));
    kernel<<<1,1>>>();
    cudaFree(foo);
}

