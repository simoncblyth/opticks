# === func-gen- : optix/optixthrust/optixthrust fgp optix/optixthrust/optixthrust.bash fgn optixthrust fgh optix/optixthrust
optixthrust-src(){      echo optix/optixthrust/optixthrust.bash ; }
optixthrust-source(){   echo ${BASH_SOURCE:-$(env-home)/$(optixthrust-src)} ; }
optixthrust-vi(){       vi $(optixthrust-source) ; }
optixthrust-env(){      elocal- ; }
optixthrust-usage(){ cat << EOU

OptiX/CUDA/Thrust Interop
==========================

TODO

* try with OpenGL backing, perhaps in glopth- 

* apply the approches/classes that come out of these investigation to ggeoview-


cmake testing
-------------

::

   optixthrust-;optixthrust-wipe;VERBOSE=1 optixthrust--



many warnings from cmake make
-------------------------------

Compilation via cmake is yielding many thousands of warnings
presumably due to the CMake kludge causing unusual compiler options.

Running verbose::

    optixthrust-
    optixthrust-wipe
    VERBOSE=1 optixthrust--
    

    -- Generating /tmp/optixthrust.cdir/lib/ptx/./OptiXThrustMinimal_generated_optixthrust_postprocess.cu.o
    /Developer/NVIDIA/CUDA-7.0/bin/nvcc \
           /Users/blyth/env/optix/optixthrust/optixthrust_postprocess.cu 
            -c -o /tmp/optixthrust.cdir/lib/ptx/./OptiXThrustMinimal_generated_optixthrust_postprocess.cu.o \
            -m64 --std c++11 \
            -Xcompiler ,\"-stdlib=libc++\",\"-mmacosx-version-min=10.8\",\"-fPIC\",\"-Wall\",\"-stdlib=libc++\",\"-O0\",\"-g3\" \
            -ccbin /usr/bin/clang \
            --use_fast_math \
             -DNVCC \
             -I/Developer/NVIDIA/CUDA-7.0/include \ 
             -I/Users/blyth/env/optix/optixthrust \
             -I/Developer/NVIDIA/CUDA-7.0/include \
             -I/Developer/OptiX/include
    /Developer/OptiX/include/optixu/optixpp_namespace.h(593): warning: overloaded virtual function "optix::APIObj::checkError" is only partially overridden in class "optix::ContextObj"

    /Developer/OptiX/include/optixu/optixpp_namespace.h(593): warning: overloaded virtual function "optix::APIObj::checkError" is only partially overridden in class "optix::ContextObj"

    /Users/blyth/env/optix/optixthrust/optixthrust_postprocess.cu:1:264: warning: unused function '__nv_init_managed_rt' [-Wunused-function]
    static char __nv_inited_managed_rt = 0 static void **__nv_fatbinhandle_for_managed_rt static void __nv_save_fatbinhandle_for_managed_rt(void **in){__nv_fatbinhandle_for_managed_rt = in} static char __nv_init_managed_rt_with_module(void **) static inline void __nv_init_managed_rt(void) { __nv_inited_managed_rt = (__nv_inited_managed_rt ? __nv_inited_managed_rt : __nv_init_managed_rt_with_module(__nv_fatbinhandle_for_managed_rt))}
                                                                                                                                                                                                                                                                           ^
    /Developer/NVIDIA/CUDA-7.0/include/channel_descriptor.h:112:37: warning: unused function 'cudaCreateChannelDescHalf' [-Wunused-function]
    static inline cudaChannelFormatDesc cudaCreateChannelDescHalf()
                                        ^
    ... 3000 unused function warnings ...


::

    simon:optixthrust blyth$ nvcc optixthrust_postprocess.cu -c -o /tmp/a.o -I$(optix-prefix)/include 
    /Developer/OptiX/include/optixu/optixpp_namespace.h(593): warning: overloaded virtual function "optix::APIObj::checkError" is only partially overridden in class "optix::ContextObj"

    /Developer/OptiX/include/optixu/optixpp_namespace.h(593): warning: overloaded virtual function "optix::APIObj::checkError" is only partially overridden in class "optix::ContextObj"

    simon:optixthrust blyth$ 

The cause is the -Wall option::

    nvcc optixthrust_postprocess.cu -c -o /tmp/a.o -I$(optix-prefix)/include -m64 --std c++11 -Xcompiler -Wall




EOU
}
optixthrust-dir(){ echo $(env-home)/optix/optixthrust ; }
optixthrust-cd(){  cd $(optixthrust-dir); }

optixthrust-env(){      
   elocal- 
   cuda-
   optix-
}

optixthrust-ptxdir(){ echo /tmp/optixthrust.ptxdir ; }
optixthrust-sdir(){   echo $(optixthrust-dir) ; }
optixthrust-bdir(){   echo /tmp/optixthrust.bdir ; }
optixthrust-cdir(){   echo /tmp/optixthrust.cdir ; }
optixthrust-idir(){   echo /tmp/optixthrust.idir ; }
optixthrust-ccd(){    cd $(optixthrust-cdir) ; }
optixthrust-bin(){    echo /tmp/optixthrust ; }

optixthrust-prep()
{
   local ptxdir=$(optixthrust-ptxdir)
   mkdir -p $ptxdir
   local bdir=$(optixthrust-bdir)
   mkdir -p $bdir 

   optixthrust-cd 
}

optixthrust-ptx-make(){
   local name=${1:-minimal_float4}
   local msg="$FUNCNAME : "
   optixthrust-prep
   local out=$(optixthrust-ptxdir)/$name.ptx
   echo $msg $name.cu to $out 
   nvcc -ptx $name.cu -o $out -I$(optix-prefix)/include
}

optixthrust-nvcc-make(){
   local msg="$FUNCNAME : "
   local name=${1:-optixthrust_postprocess}
   optixthrust-prep
   local out=$(optixthrust-bdir)/$name.cu.o
   echo $msg $name.cu to $out 
   nvcc $name.cu -c -o $out -I$(optix-prefix)/include
}

optixthrust-o-make(){
   local msg="$FUNCNAME : "
   local name=${1:-main}
   optixthrust-prep
   local out=$(optixthrust-bdir)/$name.cpp.o
   echo $msg $name.cpp to $out 

   clang $name.cpp -c -o $out \
         -I$(cuda-prefix)/include \
         -I$(optix-prefix)/include
}


optixthrust-link(){

   local objs="$*" 
   local msg="$FUNCNAME : "
   optixthrust-cd

   local bin=$(optixthrust-bin)

   echo $mss objs $objs bin $bin

   clang $objs -o $bin \
         -L$(cuda-prefix)/lib -lcudart.7.0  \
         -L$(optix-prefix)/lib64 -loptix.3.8.0 -loptixu.3.8.0  \
         -lc++ \
         -Xlinker -rpath -Xlinker $(cuda-prefix)/lib \
         -Xlinker -rpath -Xlinker $(optix-prefix)/lib64 
}

optixthrust-make-manual()
{
    optixthrust-ptx-make minimal_float4
    # not linked : loaded by OptiX at runtime

    optixthrust-nvcc-make optixthrust_postprocess

    optixthrust-o-make main

    optixthrust-o-make optixthrust

    local bdir=$(optixthrust-bdir)
    optixthrust-link $bdir/main.cpp.o $bdir/optixthrust.cu.o $bdir/optixthrust.cpp.o
}


optixthrust-run-manual(){
   local bin=$(optixthrust-bin)
   PTXDIR=$(optixthrust-ptxdir) $bin
}


optixthrust-cmake()
{
   local iwd=$PWD

   local cdir=$(optixthrust-cdir)
   mkdir -p $cdir

   optix-export
  
   optixthrust-ccd 
   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(optixthrust-idir) \
       -DOptiX_INSTALL_DIR=$(optix-install-dir) \
       -DCUDA_NVCC_FLAGS="$(optix-cuda-nvcc-flags)" \
       $(optixthrust-sdir)

   cd $iwd 

}


optixthrust-make(){
   local iwd=$PWD

   optixthrust-ccd 
   make $*

   cd $iwd 
}

optixthrust-run(){

   local cdir=$(optixthrust-cdir)
   local ptxdir=$cdir/lib/ptx
   local idir=$(optixthrust-idir)
   local ibin=$idir/bin/OptiXThrustMinimal

   PTXDIR=$ptxdir $ibin $*
}


optixthrust-wipe(){
   local cdir=$(optixthrust-cdir)
   rm -rf $cdir 
}

optixthrust--()
{
   local cdir=$(optixthrust-cdir)
   [ ! -d "$cdir" ] && optixthrust-cmake

   optixthrust-make 
   optixthrust-make install
   optixthrust-run
}

