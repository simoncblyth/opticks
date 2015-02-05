optixtest-src(){      echo optix/OptiXTest/optixtest.bash ; }
optixtest-source(){   echo ${BASH_SOURCE:-$(env-home)/$(optixtest-src)} ; }
optixtest-vi(){       vi $(optixtest-source) ; }
optixtest-env(){      
   elocal- 
   optix- 
   optix-export
}
optixtest-usage(){ cat << EOU

OptiXTest : mimimal example of OptiX
=======================================



EOU
}

optixtest-name(){ echo OptixTest ; }
optixtest-bdir(){ echo $(local-base)/env/cuda/optix/$(optixtest-name) ; }
optixtest-sdir(){ echo $(env-home)/optix/$(optixtest-name) ; }
optixtest-scd(){  cd $(optixtest-sdir); }
optixtest-bcd(){  cd $(optixtest-bdir); }
optixtest-cd(){   cd $(optixtest-sdir); }

optixtest-wipe(){
   local bdir=$(optixtest-bdir)
   rm -rf $bdir
}

optixtest-cmake(){
   local iwd=$PWD
   local bdir=$(optixtest-bdir)
   mkdir -p $bdir
   optixtest-bcd

   cmake \
         -DOptiX_INSTALL_DIR=$(optix-install-dir) \
         -DCUDA_NVCC_FLAGS="$(optix-cuda-nvcc-flags)" \
         $(optixtest-sdir)

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

