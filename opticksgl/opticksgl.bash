opticksgl-rel(){      echo opticksgl ; }
opticksgl-src(){      echo opticksgl/opticksgl.bash ; }
opticksgl-source(){   echo ${BASH_SOURCE:-$(env-home)/$(opticksgl-src)} ; }
opticksgl-vi(){       vi $(opticksgl-source) ; }
opticksgl-usage(){ cat << EOU

OpticksGL
==========

Classes depending on OptiX optixrap- and OpenGL

Classes
--------

OpViz
    High level connector between opticksop-/OpEngine and oglrap-/Scene
    with *render* method that performs OptiX ray trace and 
    pushes results to OpenGL texture

ORenderer
    connection between OpenGL renderer and OptiX ray trace

OFrame
    mechanics of OpenGL texture and OptiX buffer handling 
    and pushing from buffer to texture
    TODO: maybe dont provide public header for this


EOU
}


opticksgl-env(){      elocal- ; opticks- ; }

opticksgl-sdir(){ echo $(env-home)/opticksgl ; }

opticksgl-idir(){ echo $(opticks-idir) ; }
opticksgl-bdir(){ echo $(opticks-bdir)/$(opticksgl-rel) ; }


opticksgl-scd(){  cd $(opticksgl-sdir); }
opticksgl-cd(){   cd $(opticksgl-sdir); }

opticksgl-icd(){  cd $(opticksgl-idir); }
opticksgl-bcd(){  cd $(opticksgl-bdir); }

opticksgl-name(){ echo OpticksGL ; }

opticksgl-wipe(){
   local bdir=$(opticksgl-bdir)
   rm -rf $bdir
}

opticksgl-env(){  
   elocal- 

   optix-
   optix-export 
   optixrap-
   optixrap-export 
}

opticksgl-cmake-deprecated(){
   local iwd=$PWD

   local bdir=$(opticksgl-bdir)
   mkdir -p $bdir
  
   opticksgl-bcd 
   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(opticksgl-idir) \
       -DOptiX_INSTALL_DIR=$(optix-install-dir) \
       -DCUDA_NVCC_FLAGS="$(optix-cuda-nvcc-flags)" \
       $(opticksgl-sdir)

   cd $iwd
}

opticksgl-make(){
   local iwd=$PWD

   opticksgl-bcd 
   make $*

   cd $iwd
}

opticksgl-install(){
   opticksgl-make install
}

opticksgl--()
{
    opticksgl-make clean
    opticksgl-make
    opticksgl-install
}


