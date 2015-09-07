# === func-gen- : optix/gloptixthrust/gloptixthrust fgp optix/gloptixthrust/gloptixthrust.bash fgn gloptixthrust fgh optix/gloptixthrust
gloptixthrust-src(){      echo optix/gloptixthrust/gloptixthrust.bash ; }
gloptixthrust-source(){   echo ${BASH_SOURCE:-$(env-home)/$(gloptixthrust-src)} ; }
gloptixthrust-vi(){       vi $(gloptixthrust-source) ; }
gloptixthrust-env(){      elocal- ; }
gloptixthrust-usage(){ cat << EOU

OpenGL/OptiX/CUDA/Thrust Interop
===================================


* http://stackoverflow.com/questions/6481123/cuda-and-opengl-interop
* https://github.com/nvpro-samples/gl_cuda_interop_pingpong_st

* https://www.opengl.org/discussion_boards/showthread.php/173336-CUDA-CANNOT-perceive-changing-VBO-with-glMapBuffer



::

   gloptixthrust-;gloptixthrust-wipe;VERBOSE=1 gloptixthrust--



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
gloptixthrust-dir(){ echo $(env-home)/optix/gloptixthrust ; }
gloptixthrust-cd(){  cd $(gloptixthrust-dir); }

gloptixthrust-env(){      
   elocal- 
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




