opgl-src(){      echo opticksgl/opgl.bash ; }
opgl-source(){   echo ${BASH_SOURCE:-$(env-home)/$(opgl-src)} ; }
opgl-vi(){       vi $(opgl-source) ; }
opgl-env(){      elocal- ; }
opgl-usage(){ cat << EOU

OptiXGL
========

Classes depending OptiX optixrap- and OpenGL


EOU
}



opgl-sdir(){ echo $(env-home)/opticksgl ; }
opgl-idir(){ echo $(local-base)/env/opticksgl ; }
opgl-bdir(){ echo $(opgl-idir).build ; }

opgl-scd(){  cd $(opgl-sdir); }
opgl-cd(){   cd $(opgl-sdir); }

opgl-icd(){  cd $(opgl-idir); }
opgl-bcd(){  cd $(opgl-bdir); }

opgl-name(){ echo OpticksGL ; }

opgl-wipe(){
   local bdir=$(opgl-bdir)
   rm -rf $bdir
}

opgl-env(){  
   elocal- 

   optix-
   optix-export 
   optixrap-
   optixrap-export 
}

opgl-cmake(){
   local iwd=$PWD

   local bdir=$(opgl-bdir)
   mkdir -p $bdir
  
   opgl-bcd 
   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(opgl-idir) \
       -DOptiX_INSTALL_DIR=$(optix-install-dir) \
       -DCUDA_NVCC_FLAGS="$(optix-cuda-nvcc-flags)" \
       $(opgl-sdir)

   cd $iwd
}

opgl-make(){
   local iwd=$PWD

   opgl-bcd 
   make $*

   cd $iwd
}

opgl-install(){
   opgl-make install
}

opgl--()
{
    opgl-cmake
    opgl-make
    opgl-install
}



