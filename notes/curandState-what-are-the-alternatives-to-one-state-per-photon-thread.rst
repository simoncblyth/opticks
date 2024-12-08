curandState-what-are-the-alternatives-to-one-state-per-photon-thread
========================================================================


curand docs
----------------

* https://docs.nvidia.com/cuda/curand/device-api-overview.html#quasirandom-sequences


my curand_init
-----------------

::

     08 /**
      9 _QCurandState_curand_init_chunk
     10 ---------------------------------
     11 
     12 id 
     13    [0:threads_per_launch]
     14 
     15 states_thread_offset 
     16    enables multiple launches to write into the correct output slot
     17 
     18 **/
     19 
     20 
     21 __global__ void _QCurandState_curand_init_chunk(int threads_per_launch, int id_offset, scurandref* cr, curandState* states_thread_offset )
     22 {
     23     int id = blockIdx.x*blockDim.x + threadIdx.x;
     24     if (id >= threads_per_launch) return;
     25     curand_init(cr->seed, id+id_offset, cr->offset, states_thread_offset + id );
     26 
     27     //if( id == 0 ) printf("// _QCurandState_curand_init_chunk id_offset %d \n", id_offset ); 
     28 }





GPU stateless RNG
-------------------



:google:`curand generator with smallest state`
-------------------------------------------------

* https://stackoverflow.com/questions/37362554/minimize-the-number-of-curand-states-stored-during-a-mc-simulation

crovella + talonmies suggest to reuse curandState within the thread, so can have less of them.



* https://forums.developer.nvidia.com/t/random-number-generation-using-curand-curandstatephilox4-32-10-t-use-local-registers-or-shared/39439/4



* https://softwareengineering.stackexchange.com/questions/429454/how-does-curand-use-a-gpu-to-accelerate-random-number-generation-dont-those-re


curand-done-right : interesting because it builds ontop of curand_Philox4x32_10 rather minimally
-----------------------------------------------------------------------------------------------------

* https://github.com/kshitijl/curand-done-right/blob/master/README.md


::

    001 #pragma once
      2 
      3 #include <curand_kernel.h>
      4 #include <curand_normal.h>
      5 
      6 namespace curanddr {
      7   template <int Arity, typename num_t = float>
      8   struct alignas(8) vector_t {
      9     num_t values[Arity];
     10     __device__ num_t operator[] (size_t n) const {
     11       return values[n];
     12     }
     13   };
     14 
     15   template<typename num_t>
     16   struct alignas(8) vector_t<1, num_t> {
     17     num_t values[1];
     18     __device__ num_t operator[] (size_t n) const {
     19       return values[n];
     20     }
     21     __device__ operator num_t() const {
     22       return values[0];
     23     }
     24   };
     25 
     26   // from moderngpu meta.hxx
     27   template<int i, int count, bool valid = (i < count)>
     28   struct iterate_t {
     29     template<typename func_t>
     30     __device__ static void eval(func_t f) {
     31       f(i);
     32       iterate_t<i+1, count>::eval(f);
     33     }
     34   };
     35 
     36   template<int i, int count>
     37   struct iterate_t<i, count, false> {
     38     template<typename func_t>
     39     __device__ static void eval(func_t f) { }
     40   };
     41 
     42   template<int count, typename func_t>
     43   __device__ void iterate(func_t f) {
     44     iterate_t<0, count>::eval(f);
     45   }
    ...

    076   template<int Arity>
     77   __device__ vector_t<Arity> uniforms(uint4 counter, uint key) {
     78     enum { n_blocks = (Arity + 4 - 1)/4 };
     79 
     80     float scratch[n_blocks * 4];
     81  
     82     iterate<n_blocks>([&](uint index) {
     83         uint2 local_key{key, index};
     84         uint4 result = curand_Philox4x32_10(counter, local_key);
     85 
     86         uint ii = index*4;
     87         scratch[ii]   = _curand_uniform(result.x);
     88         scratch[ii+1] = _curand_uniform(result.y);
     89         scratch[ii+2] = _curand_uniform(result.z);
     90         scratch[ii+3] = _curand_uniform(result.w);
     91       });
     92 
     93     vector_t<Arity> answer;
     94 
     95     iterate<Arity>([&](uint index) {
     96         answer.values[index] = scratch[index];
     97       });
     98  
     99     return answer;
    100   }



    P[blyth@localhost include]$ grep curand_Philox *.h
    curand_kernel.h:        state->output = curand_Philox4x32_10(state->ctr,state->key);
    curand_kernel.h:    state->output= curand_Philox4x32_10(state->ctr,state->key);
    curand_kernel.h:    state->output = curand_Philox4x32_10(state->ctr,state->key);
    curand_kernel.h:    state->output = curand_Philox4x32_10(state->ctr,state->key);
    curand_philox4x32_x.h:QUALIFIERS uint4 curand_Philox4x32_10( uint4 c, uint2 k)
    P[blyth@localhost include]$ 
    P[blyth@localhost include]$ 


/usr/local/cuda/include/curand_philox4x32_x.h







::

    P[blyth@localhost curand-done-right]$ bin/basic-pi-example 100
    85 85
    3.400000
    P[blyth@localhost curand-done-right]$ bin/basic-pi-example 1000
    792 792
    3.168000
    P[blyth@localhost curand-done-right]$ bin/basic-pi-example 10000
    7878 621
    3.151200
    P[blyth@localhost curand-done-right]$ bin/basic-pi-example 100000
    78425 527
    3.137000
    P[blyth@localhost curand-done-right]$ bin/basic-pi-example 1000000
    785338 448
    3.141352
    P[blyth@localhost curand-done-right]$ bin/basic-pi-example 10000000
    7855546 502
    3.142218
    P[blyth@localhost curand-done-right]$ bin/basic-pi-example 100000000
    78535604 196
    3.141424
    P[blyth@localhost curand-done-right]$ bin/basic-pi-example 1000000000
    785411250 411
    3.141645
    P[blyth@localhost curand-done-right]$ 





:google:`curand Random123`
-------------------------------

* https://github.com/lemire/random123/tree/master
* https://github.com/lemire/random123/blob/master/include/Random123/philox.h

* https://www.thesalmons.org/john/random123/releases/1.00/docs/group__PhiloxNxW.html


Detailed Description

The PhiloxNxW classes export the member functions, typedefs and operator
overloads required by a CBRNG class.

As described in Parallel Random Numbers: As Easy as 1, 2, 3 . The Philox family
of counter-based RNGs use integer multiplication, xor and permutation of
word-sized blocks to scramble the bits of its input key. Philox is a mnemonic
for Product HI LO Xor). 

* https://numpy.org/doc/2.0/reference/random/bit_generators/philox.html



Vectorization of random number generation
--------------------------------------------

Vectorization of random number generation and
reproducibility of concurrent particle transport
simulation
To cite this article: S Y Jun et al 2020
J. Phys.: Conf. Ser. 1525 01205

* https://iopscience.iop.org/article/10.1088/1742-6596/1525/1/012054/pdf


NEST
-----

* https://nest-simulator.readthedocs.io/en/v3.0/guides/nest2_to_nest3/nest3_features/random_number_generators.html

CBPRNG
---------

https://en.wikipedia.org/wiki/Counter-based_random_number_generator

A counter-based random number generation (CBRNG, also known as a counter-based
pseudo-random number generator, or CBPRNG) is a kind of pseudorandom number
generator that uses only an integer counter as its internal state. They are
generally used for generating pseudorandom numbers for large parallel
computations. 


CBRNGs based on multiplication

In addition to Threefry and ARS, Salmon et al. described a third counter-based
PRNG, Philox,[1] based on wide multiplies; e.g. multiplying two 32-bit numbers
and producing a 64-bit number, or multiplying two 64-bit numbers and producing
a 128-bit number.

As of 2020, Philox is popular on CPUs and GPUs. On GPUs, nVidia's cuRAND
library[5] and TensorFlow[6] provide implementations of Philox. On CPUs,
Intel's MKL provides an implementation.

A new CBRNG based on multiplication is the Squares RNG.[7] This generator
passes stringent tests of randomness[8] and is considerably faster than Philox. 


Squares: A Fast Counter-Based RNG
-------------------------------------

Widynski, Bernard (2020). "Squares: A Fast Counter-Based RNG"
* https://arxiv.org/abs/2004.06278
* squaresrng.wixsite.com/rand

philox
-------

* https://github.com/dsnz/random/blob/master/philox.py



Random123: a Library of Counter-Based Random Number Generators
---------------------------------------------------------------

* http://www.thesalmons.org/john/random123/releases/1.11.2pre/docs/



CBPRNG : counter-based pseudorandom number generators
----------------------------------------------------------

* https://www.epj-conferences.org/articles/epjconf/pdf/2021/05/epjconf_chep2021_03039.pdf

This document describes how the issues discussed in the previous paragraphs have been
overcome in CORSIKA 8, via the deployment of “counter-based pseudorandom number gen-
erators” (CBPRNGs) and their management through an iterator-based, STL compliant, and
parallelism enabled interface. 




OpenRAND
~~~~~~~~~~

* https://github.com/msu-sparta/OpenRAND
* https://arxiv.org/pdf/2310.19925

OpenRAND: A Performance Portable, Reproducible Random Number Generation Library
for Parallel Computations
Shihab Shahriar Khana,∗, Bryce Palmerb,c, Christopher Edelmaierd, Hasan Metin Aktulga




* https://www.epj-conferences.org/articles/epjconf/pdf/2024/05/epjconf_chep2024_11001.pdf

A new portable random number generator wrapper library
Tianle Wang1,∗, Mohammad Atif1, Zhihua Dong1, Charles Leggett2, and Meifeng Lin1

BUT: I  DIDNT FIND THE REPO ? 



* https://www.thesalmons.org/john/random123/papers/random123sc11.pdf


Parallel Random Numbers: As Easy as 1, 2, 3
John K. Salmon,∗ Mark A. Moraes, Ron O. Dror, and David E. Shaw∗†
D. E. Shaw Research, New York, NY 10036, USUT: I think that would 

Ignoring some details (see Section 2 for a
complete deﬁnition), the sequence is:
xn = b(n). (2)
The nth random number is obtained directly by some func-
tion, b, applied to n. In the simplest case, n is a p-bit in-
teger counter, so we call this class of PRNGs counter-based.
Equation 2 is inherently parallel; that is, it eliminates the
sequential dependence between successive xn in Eqn 1


* https://github.com/pytorch/pytorch/issues/263



:google:`curand without state`
-------------------------------



* https://forums.developer.nvidia.com/t/curand-my-implementation-works-but-i-am-not-sure-its-the-right-way-to-do-it/176128/2


Richard Harris : grid stride loops
---------------------------------------

* https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/



::

    void saxpy(int n, float a, float *x, float *y)
    {
        for (int i = 0; i < n; ++i)
            y[i] = a * x[i] + y[i];
    }



::

    __global__
    void saxpy(int n, float a, float *x, float *y)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) 
            y[i] = a * x[i] + y[i];
    }





Instead of completely eliminating the loop when parallelizing the computation, 
I recommend to use a grid-stride loop, as in the following kernel.::

    __global__
    void saxpy(int n, float a, float *x, float *y)
    {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
             i < n; 
             i += blockDim.x * gridDim.x) 
          {
              y[i] = a * x[i] + y[i];
          }
    }



Rather than assume that the thread grid is large enough to cover the entire
data array, this kernel loops over the data array one grid-size at a time.

Notice that the stride of the loop is blockDim.x * gridDim.x which is the total
number of threads in the grid. So if there are 1280 threads in the grid, thread
0 will compute elements 0, 1280, 2560, etc. This is why I call this a
grid-stride loop. By using a loop with stride equal to the grid size, we ensure
that all addressing within warps is unit-stride, so we get maximum memory
coalescing, just as in the monolithic version.






