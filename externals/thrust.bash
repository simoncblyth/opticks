##
## Copyright (c) 2019 Opticks Team. All Rights Reserved.
##
## This file is part of Opticks
## (see https://bitbucket.org/simoncblyth/opticks).
##
## Licensed under the Apache License, Version 2.0 (the "License"); 
## you may not use this file except in compliance with the License.  
## You may obtain a copy of the License at
##
##   http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software 
## distributed under the License is distributed on an "AS IS" BASIS, 
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
## See the License for the specific language governing permissions and 
## limitations under the License.
##

thrust-source(){   echo ${BASH_SOURCE} ; }
thrust-vi(){       vi $(thrust-source) ; }
thrust-usage(){ cat << EOU

Thrust
=======

The Thrust that comes with CUDA is used for Opticks, 
no separate install needed. Any gets done below
are just for the documentation and source/examples perusal.


Examples
---------

* https://github.com/thrust/thrust/tree/master/examples

Refs
-----

* http://thrust.github.io

* https://developer.nvidia.com/Thrust

* https://code.google.com/p/thrust/wiki/QuickStartGuide

* https://developer.nvidia.com/gpu-accelerated-libraries

* http://astronomy.swin.edu.au/supercomputing/thrust.pdf


HEP Thrust Usage
------------------


* https://github.com/MultithreadCorner/MCBooster


Experts
--------

* http://stackoverflow.com/users/1695960/robert-crovella

Thrust sync
-----------

* :google:`thrust cudaStream`

* https://github.com/thrust/thrust/issues/664

* http://devblogs.nvidia.com/parallelforall/gpu-pro-tip-cuda-7-streams-simplify-concurrency/

* http://www.techenablement.com/the-cuda-thrust-api-now-supports-streams-and-concurrent-tasks/



Heterogenous Class Definition 
-----------------------------

By heterogenous classes, I mean ones that combine methods 
compiled with nvcc and the underlying system compiler eg clang, gcc.
The alternative to doing this is split up the CPU and GPU code into 
separate 
and do the work of providing 

 
Was initially surprised to find that this works, but on reflection of 
what *nvcc* actually does it makes sense.  
*nvcc* is a compiler driver that splits code up into portions 
to run on CPU which are compiled with the underlying compiler 
and portions to be compiled for the GPU.

Because *nvcc* is using the undelying  

hemi
~~~~~

* http://devblogs.nvidia.com/parallelforall/developing-portable-cuda-cc-code-hemi/


Thrust CUDA Interop
----------------------

See env/numerics/thrust/thrust_cuda_interop

Thrust Functor Access to device arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://stackoverflow.com/questions/27733652/how-can-i-dereference-a-thrustdevice-vector-from-within-a-thrust-functor


Linking Thrust with non-Thrust code
------------------------------------

Need to do gymnastics to avoid nvcc specifics from being in headers
eg cannot include a thrust functor in a header that needs to 
be compiled by clang/gcc.


Define them away

* https://groups.google.com/forum/#!topic/thrust-users/abVI3htMrkw
  
::

    // CudaPortability.h
    // Allow host/device functions to be directly included in pure cpp
    #ifdef __CUDACC__
    #  define CUDA_PROTOTYPE   __device__ __host__
    #  define CUDA_INLINE      __device__ __host__ inline
    #else
    #  define CUDA_PROTOTYPE
    #  define CUDA_INLINE      inline
    #endif


Thrust defines them away for non nvcc in /Developer/NVIDIA/CUDA-7.0/include/thrust/detail/config/host_device.h 
::

   #include <thrust/detail/config/host_device.h>



Involved Thrust
----------------

* http://stackoverflow.com/questions/26666621/thrust-fill-isolate-space


Parallel Processing Algorthms Intro, eg explaining Scatter/Gather
-------------------------------------------------------------------

* http://heather.cs.ucdavis.edu/~matloff/158/PLN/ParProcBook.pdf#page161


Thrust cmake
-------------

* http://stackoverflow.com/questions/28968277/compilation-error-using-findcuda-cmake-and-thrust-with-thrust-device-system-omp
* http://stackoverflow.com/questions/13073717/building-cuda-object-files-using-cmake

Thrust OptiX interop
--------------------

* https://devtalk.nvidia.com/search/more/sitecommentsearch/optix%20thrust/

* https://github.com/thrust/thrust/issues/204

  Using thrust with some OptiX types

* https://devtalk.nvidia.com/default/topic/574078/optix/compiler-errors-when-including-both-optix-and-thrust/

Thrust CUDA interop
--------------------

* https://github.com/thrust/thrust/blob/master/examples/cuda/wrap_pointer.cu

Frequency Indexing, Histogramming
-----------------------------------

* http://stackoverflow.com/questions/8792926/finding-the-number-of-occurrences-of-keys-and-the-positions-of-first-occurrences
* https://code.google.com/p/thrust/source/browse/examples/histogram.cu

Thrust Long Long
------------------

* 
* https://github.com/thrust/thrust/issues/658 

Thrust Memory Access Patterns, functor state
-----------------------------------------------


Memory access pattern is what matters

* http://stackoverflow.com/questions/17260207/cuda-how-does-thrust-manage-memory-when-using-a-comparator-in-a-sorting-functio

Accessing constant memory from thrust functor

* http://stackoverflow.com/questions/17064096/thrustdevice-vector-in-constant-memory


Thrust FAQ
------------

* https://github.com/thrust/thrust/wiki/Frequently-Asked-Questions

Can I create a thrust::device_vector from memory I've allocated myself?
No. Instead, wrap your externally allocated raw pointer with thrust::device_ptr 
and pass it to Thrust algorithms.

* https://github.com/thrust/thrust/blob/master/examples/cuda/wrap_pointer.cu

Limitations ?
~~~~~~~~~~~~~~

Any limitations of using thrust::device_ptr<T> compared to thrust::device_vector<T>




EOU
}


thrust-edir(){ echo $(opticks-home)/numerics/thrust ; }
thrust-sdir(){ echo $(local-base)/env/numerics/thrust ; }
thrust-idir(){ echo $(cuda-idir)/thrust ; }

thrust-ecd(){  cd $(thrust-edir) ; }
thrust-scd(){  cd $(thrust-sdir) ; }
thrust-icd(){  cd $(thrust-idir) ; }

thrust-cd(){   cd $(thrust-idir) ; }




thrust-get()
{
    # not for building, just for the examples
    local dir=$(dirname $(thrust-sdir)) &&  mkdir -p $dir && cd $dir
    git clone https://github.com/thrust/thrust.git
}

thrust-update()
{
    thrust-scd
    git pull
}


thrust-env(){      
   olocal- ; 
   cuda- ; 
}

thrust-cuda-nvcc-flags(){
   echo -ccbin /usr/bin/clang --use_fast_math
}

thrust-export()
{
   echo -n
}


thrust-samples-dir(){ echo $(cuda-dir)/samples ; }
thrust-samples-cd(){  cd $(thrust-samples-dir) ; }

thrust-samples-find(){ 
   thrust-samples-cd
   find . -type f -exec grep -l thrust {} \;
}


thrust-pdf(){  open $(cuda-dir)/doc/pdf/Thrust_Quick_Start_Guide.pdf ; }
thrust-html(){ open $(cuda-dir)/doc/html/thrust/index.html ; }

thrust-doc(){  open https://github.com/thrust/thrust/wiki/Documentation ; }

thrust-ex(){ open https://github.com/thrust/thrust/tree/master/examples ; }




