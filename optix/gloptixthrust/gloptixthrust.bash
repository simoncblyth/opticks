# === func-gen- : optix/gloptixthrust/gloptixthrust fgp optix/gloptixthrust/gloptixthrust.bash fgn gloptixthrust fgh optix/gloptixthrust
gloptixthrust-src(){      echo optix/gloptixthrust/gloptixthrust.bash ; }
gloptixthrust-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(gloptixthrust-src)} ; }
gloptixthrust-vi(){       vi $(gloptixthrust-source) ; }
gloptixthrust-env(){      olocal- ; }
gloptixthrust-usage(){ cat << EOU

OpenGL/OptiX/CUDA/Thrust Interop
===================================

Interop Objectives
-------------------

* minimize host allocations
* minimize duplication on the device
 
Production : non-OpenGL OptiX/Thrust interop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* create OptiX buffers, populate with OptiX launch
* stream compaction pullback buffer subsets based on criteria
  from another buffer ?  
 
Debug : OpenGL/OptiX/Thrust interop (possibly pairwise)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* createFromGLBO OptiX buffers, populate with OptiX launch 

Huh
----

* http://stackoverflow.com/questions/11692326/can-i-use-thrusthost-vector-or-i-must-use-cudahostalloc-for-zero-copy-with-thr

Related
--------

* optixthrust- 

What Next
----------

* tidyup/rename classes for reusability and move into optixrap- thrustrap-

* review thrustrap- optixrap- ggeoview- for working out how to bring in 
  new interop/thrust understanding and remove old attempts 

  * thrustrap- makes heavy use of device_vector where perhaps device_pointer is needed 
    to avoid copies

    * https://github.com/thrust/thrust/blob/master/examples/uninitialized_vector.cu

  * ggeoview- defer large NPY allocations until actually needed


Refs
----

* http://stackoverflow.com/questions/6481123/cuda-and-opengl-interop
* https://github.com/nvpro-samples/gl_cuda_interop_pingpong_st

* https://www.opengl.org/discussion_boards/showthread.php/173336-CUDA-CANNOT-perceive-changing-VBO-with-glMapBuffer

::

   gloptixthrust-;gloptixthrust-wipe;VERBOSE=1 gloptixthrust--



OpenGL/Thrust Interop Issue
-----------------------------

* cudaMemcpy, cudaMemset and kernel calls work with interop as expected
* thrust::transform/copy etc calls do not 

  * **CORRECTION : YES THEY DO**
  * CURRENTLY THIS LOOKS TO HAVE BEEN A MISUNDERSTANDING REGARDS DEVICE_VECTOR CTOR

This suggests the problem is with Thrust or my use of Thrust rather
than the Interop mapping/unmapping.

Only one mention of OpenGL in Thrust issues

* https://github.com/thrust/thrust/issues/683

::

     thrust::sort_by_key(thrust::cuda::par.on(stream), dev_keys, dev_keys + N, dev_values);


* :google:`thrust::cuda::par.on`

  * https://github.com/thrust/thrust/wiki/Direct-System-Access  


Thrust Stream Issue
--------------------

* http://devblogs.nvidia.com/parallelforall/gpu-pro-tip-cuda-7-streams-simplify-concurrency/
* https://github.com/thrust/thrust/issues/664


CUDA Samples : Graphics
-------------------------

* http://docs.nvidia.com/cuda/cuda-samples/index.html#graphics

Searching for Thrust on the above pages, finds a few with interop

* http://docs.nvidia.com/cuda/cuda-samples/index.html#marching-cubes-isosurfaces
* http://docs.nvidia.com/cuda/cuda-samples/index.html#smoke-particles


Thrust OpenGL interop
----------------------

* http://stackoverflow.com/questions/24966586/sorting-pixels-from-opengl-using-cuda-and-thrust
* needs modification wrt opengl headers : uses both kernel call and thrust funcs


Runtime vs Driver Interop ?
----------------------------

Runtime, *cuda*  API::

   #include <cuda_gl_interop.h>

Driver, *cu* API (more direct control)::

   #include <cudagl.h>


CUDA kernel call with interop
-------------------------------

* http://research.ncl.ac.uk/game/mastersdegree/gametechnologies/cudatutorial3cuda-openglinteroperability/CUDA%203.pdf
* http://www.ecse.rpi.edu/~wrf/wiki/ParallelComputingSpring2014/cuda-by-example/chapter08/basic_interop.cu 




Thrust async 
-------------

* http://stackoverflow.com/questions/11116750/is-thrust-synchronous-or-asynchronous


Kernel launches have always been asynchronous - even in CUDA 1.0 - so any
Thrust call that results only in a kernel launch will be asynchronous.

Any Thrust code that implicitly triggers memcpy's will be synchronous due
to the lack of stream support, 

For example, thrust::reduce() is definitely synchronous since it reads back the
result and returns it to the calling thread via the return value.

* http://devblogs.nvidia.com/parallelforall/expressive-algorithmic-programming-thrust/




EOU
}
gloptixthrust-dir(){ echo $(opticks-home)/optix/gloptixthrust ; }
gloptixthrust-cd(){  cd $(gloptixthrust-dir); }

gloptixthrust-env(){      
   olocal- 
   cuda-
   optix-
}

gloptixthrust-sdir(){   echo $(gloptixthrust-dir) ; }
gloptixthrust-ptxdir(){ echo /tmp/gloptixthrust.ptxdir ; }
gloptixthrust-bdir(){   echo /tmp/gloptixthrust.bdir ; }
gloptixthrust-cdir(){   echo /tmp/gloptixthrust.cdir ; }
gloptixthrust-idir(){   echo /tmp/gloptixthrust.idir ; }
gloptixthrust-ccd(){    cd $(gloptixthrust-cdir) ; }

gloptixthrust-cmake()
{
   local iwd=$PWD

   local cdir=$(gloptixthrust-cdir)
   mkdir -p $cdir

   optix-export
  
   gloptixthrust-ccd 
   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(gloptixthrust-idir) \
       -DOptiX_INSTALL_DIR=$(optix-install-dir) \
       -DCUDA_NVCC_FLAGS="$(optix-cuda-nvcc-flags)" \
       $(gloptixthrust-sdir)

   cd $iwd 

}


gloptixthrust-make(){
   local iwd=$PWD

   gloptixthrust-ccd 
   make $*

   cd $iwd 
}

gloptixthrust-run(){

   local cdir=$(gloptixthrust-cdir)
   local ptxdir=$cdir/lib/ptx
   local idir=$(gloptixthrust-idir)
   local ibin=$idir/bin/GLOptiXThrustMinimal

   PTXDIR=$ptxdir $LLDB $ibin $*
}

gloptixthrust-lldb(){
   LLDB=lldb gloptixthrust-run $*
}

gloptixthrust-oac(){
   OPTIX_API_CAPTURE=1 gloptixthrust-run $*
}

gloptixthrust-wipe(){
   local cdir=$(gloptixthrust-cdir)
   rm -rf $cdir 
}

gloptixthrust--()
{
   local cdir=$(gloptixthrust-cdir)
   [ ! -d "$cdir" ] && gloptixthrust-cmake

   gloptixthrust-make 
   gloptixthrust-make install
   gloptixthrust-run
}




