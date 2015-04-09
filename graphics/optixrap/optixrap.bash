# === func-gen- : graphics/optixrap/optixrap fgp graphics/optixrap/optixrap.bash fgn optixrap fgh graphics/optixrap
optixrap-src(){      echo graphics/optixrap/optixrap.bash ; }
optixrap-source(){   echo ${BASH_SOURCE:-$(env-home)/$(optixrap-src)} ; }
optixrap-vi(){       vi $(optixrap-source) ; }
optixrap-usage(){ cat << EOU

Hmm OptiX CMakeLists are kinda compilicated making it difficult 
to do an equvalent to oglrap- but maybe some of the
component needed can be stuffed into a library without the 
full CMakeLists machinery for compiling .cu to .ptx etc..


EOU
}


optixrap-sdir(){ echo $(env-home)/graphics/optixrap ; }
optixrap-idir(){ echo $(local-base)/env/graphics/optixrap ; }
optixrap-bdir(){ echo $(optixrap-idir).build ; }

optixrap-scd(){  cd $(optixrap-sdir); }
optixrap-cd(){   cd $(optixrap-sdir); }

optixrap-icd(){  cd $(optixrap-idir); }
optixrap-bcd(){  cd $(optixrap-bdir); }
optixrap-name(){ echo OptiXRap ; }

optixrap-wipe(){
   local bdir=$(optixrap-bdir)
   rm -rf $bdir
}

optixrap-env(){  
   elocal- 
   optix-
   optix-export 
}

optixrap-cmake(){
   local iwd=$PWD

   local bdir=$(optixrap-bdir)
   mkdir -p $bdir
  
   optixrap-bcd 
   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(optixrap-idir) \
       -DOptiX_INSTALL_DIR=$(optix-install-dir) \
       -DCUDA_NVCC_FLAGS="$(optix-cuda-nvcc-flags)" \
       $(optixrap-sdir)

   cd $iwd
}


optixrap-make(){
   local iwd=$PWD

   optixrap-bcd 
   make $*

   cd $iwd
}

optixrap-install(){
   optixrap-make install
}

optixrap-bin(){ echo $(optixrap-idir)/bin/$(optixrap-name)Test ; }
optixrap-export()
{
   export SHADER_DIR=$(optixrap-sdir)/glsl
}
optixrap-run(){
   local bin=$(optixrap-bin)
   optixrap-export
   $bin $*
}



optixrap--()
{
    optixrap-wipe
    optixrap-cmake
    optixrap-make
    optixrap-install

}







