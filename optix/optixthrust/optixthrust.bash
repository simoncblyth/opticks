# === func-gen- : optix/optixthrust/optixthrust fgp optix/optixthrust/optixthrust.bash fgn optixthrust fgh optix/optixthrust
optixthrust-src(){      echo optix/optixthrust/optixthrust.bash ; }
optixthrust-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(optixthrust-src)} ; }
optixthrust-vi(){       vi $(optixthrust-source) ; }
optixthrust-env(){      elocal- ; }
optixthrust-usage(){ cat << EOU

OptiX/CUDA/Thrust Interop
==========================

The *optixthrust-* package is a testing ground for OptiX/CUDA/Thrust interop,
without any *OpenGL* complications.

see also
---------

* gloptixthrust- 
* optixminimal- 
* glfwtriangle-



CMAKE CUDA/OptiX Libraries
----------------------------

OptiX ptx and vanilla OBJ all go to CUDA_GENERATED_OUTPUT_DIR when that is defined, 
which is messy.

OptiX FindCUDA::

    1268       # Determine output directory
    1269       cuda_compute_build_path("${file}" cuda_build_path)
    1270       set(cuda_compile_intermediate_directory "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${cuda_target}.dir/${cuda_build_path}")
    1271       if(CUDA_GENERATED_OUTPUT_DIR)
    1272         set(cuda_compile_output_dir "${CUDA_GENERATED_OUTPUT_DIR}")
    1273       else()
    1274         if ( compile_to_ptx )
    1275           set(cuda_compile_output_dir "${CMAKE_CURRENT_BINARY_DIR}")
    1276         else()
    1277           set(cuda_compile_output_dir "${cuda_compile_intermediate_directory}")
    1278         endif()
    1279       endif()
     




investigations
---------------

* stream compaction
* multiple argument transform functors, using tuples

cmake testing
-------------

::

   optixthrust-;optixthrust-wipe;VERBOSE=1 optixthrust--


interesting snippet on stream compaction
------------------------------------------

* https://github.com/thrust/thrust/blob/master/examples/stream_compaction.cu
* https://github.com/thrust/thrust/issues/204

::

    // thrust-optix interop cmake build testing
    //  https://github.com/thrust/thrust/issues/204

    #include <thrust/device_vector.h> 
    #include <thrust/remove.h> 

    namespace optix { 
       class __align__(16) Aabb { 
          float3 m_min; 
          float3 m_max; 
       };  
    }// end namespace optix 


    template<typename T > 
    struct isZero { 
        __host__ __device__ T operator()(const T &x) const {return x==0;} 
    }; 

    void aabbValidCompaction(optix::Aabb *boxes, unsigned int *stencil, size_t num) 
    { 
        thrust::device_ptr<optix::Aabb > dev_begin_ptr(boxes); 
        thrust::device_ptr<optix::Aabb > dev_end_ptr(boxes + num); 
        thrust::device_ptr<unsigned int > dev_stencil_ptr(stencil); 
        thrust::remove_if(dev_begin_ptr, dev_end_ptr, dev_stencil_ptr, isZero<unsigned int >()); 
    } 


CUDA struct align for passing photons to thrust functors
------------------------------------------------------------

* http://stackoverflow.com/questions/12778949/cuda-memory-alignment

::

    #if defined(__CUDACC__) // NVCC
       #define MY_ALIGN(n) __align__(n)
    #elif defined(__GNUC__) // GCC
      #define MY_ALIGN(n) __attribute__((aligned(n)))
    #elif defined(_MSC_VER) // MSVC
      #define MY_ALIGN(n) __declspec(align(n))
    #elif defined(__clang__) // 
      #define MY_ALIGN(n) __attribute__((aligned(n)))
    #else
      #error "Please provide a definition for MY_ALIGN macro for your host compiler!"
    #endif

::

    struct MY_ALIGN(16) pt { int i, j, k; }
    # enforces that the memory for the struct begins at an address in memory that is a multiple of n bytes


* float4 is 4 * 4 = 16 bytes 

::

    typedef struct MY_ALIGN(16) photon_t ;
    photon_t photon { float4 a,b,c,d ; }


passing multiple float4 to thrust functors using thrust tuple
---------------------------------------------------------------

/usr/local/env/cuda/NVIDIA_CUDA-7.0_Samples/5_Simulations/smokeParticles/ParticleSystem_cuda.cu
/usr/local/env/cuda/NVIDIA_CUDA-7.0_Samples/5_Simulations/smokeParticles/particles_kernel_device.cuh


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
optixthrust-dir(){ echo $(opticks-home)/optix/optixthrust ; }
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
optixthrust-idir(){   echo $(local-base)/env/graphics/optixthrust ; }
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
   $bin
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

   #local cdir=$(optixthrust-cdir)
   #local ptxdir=$cdir/lib/ptx
   local idir=$(optixthrust-idir)
   local ibin=$idir/bin/OptiXThrustMinimal

   #PTXDIR=$ptxdir $ibin $*
   $ibin $*
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
   #optixthrust-run
}

