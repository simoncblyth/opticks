# === func-gen- : cuda/optix/OptiXTest/optixtest fgp cuda/optix/OptiXTest/optixtest.bash fgn optixtest fgh cuda/optix/OptiXTest
optixtest-src(){      echo cuda/optix/OptiXTest/optixtest.bash ; }
optixtest-source(){   echo ${BASH_SOURCE:-$(env-home)/$(optixtest-src)} ; }
optixtest-vi(){       vi $(optixtest-source) ; }
optixtest-env(){      
   elocal- 
   optix- 
   optix-export
}
optixtest-usage(){ cat << EOU


::

    INFOnvcc_host_compiler_flags = ""
    /Users/blyth/env/cuda/optix/OptixTest/draw_color.cu(13): Warning: Cannot tell what pointer points to, assuming global memory space

EOU
}

optixtest-name(){ echo OptixTest ; }
optixtest-bdir(){ echo $(local-base)/env/cuda/optix/$(optixtest-name) ; }
optixtest-sdir(){ echo $(env-home)/cuda/optix/$(optixtest-name) ; }
optixtest-scd(){  cd $(optixtest-sdir); }
optixtest-bcd(){  cd $(optixtest-bdir); }
optixtest-cd(){   cd $(optixtest-sdir); }

optixtest-cmake(){
   local iwd=$PWD
   local bdir=$(optixtest-bdir)
   mkdir -p $bdir
   optixtest-bcd

   cmake -DOptiX_INSTALL_DIR=$(optix-install-dir) -DCUDA_NVCC_FLAGS="-ccbin /usr/bin/clang --use_fast_math" $(optixtest-sdir)

   cd $iwd
}

optixtest-make(){
   local iwd=$PWD
   optixtest-bcd
   make $*
   cd $iwd
}

optixtest-run(){
   local cmd="$(optixtest-bdir)/$(optixtest-name)"
   echo $cmd
   eval $cmd
}

