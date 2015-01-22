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





EOU
}
optixtest-bdir(){ echo $(local-base)/env/cuda/optix/OptiXTest ; }
optixtest-sdir(){ echo $(env-home)/cuda/optix/OptiXTest ; }
optixtest-scd(){  cd $(optixtest-sdir); }
optixtest-bcd(){  cd $(optixtest-bdir); }
optixtest-cd(){   cd $(optixtest-sdir); }

optixtest-cmake(){
   local bdir=$(optixtest-bdir)
   mkdir -p $bdir
   optixtest-bcd

   cmake -DOptiX_INSTALL_DIR=$(optix-install-dir) $(optixtest-sdir)
}

optixtest-make(){
   optixtest-bcd
   make $*
}
