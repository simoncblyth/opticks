# === func-gen- : graphics/oglrap/oglrap fgp graphics/oglrap/oglrap.bash fgn oglrap fgh graphics/oglrap
oglrap-src(){      echo graphics/oglrap/oglrap.bash ; }
oglrap-source(){   echo ${BASH_SOURCE:-$(env-home)/$(oglrap-src)} ; }
oglrap-vi(){       vi $(oglrap-source) ; }
oglrap-env(){      elocal- ; }
oglrap-usage(){ cat << EOU

Featherweight OpenGL wrapper
==============================

Just a few utility classes to make modern OpenGL 3, 4 
easier to use.

Originally was thinking could keep GLFW3 out of oglrap
keeping pure OpenGL, but that has proved difficult.

GLFW3 and GLEW are responsible for furnishing the 
OpenGL headers, but wanted to avoid directly using 
GLFW3 inside oglrap-




EOU
}


oglrap-sdir(){ echo $(env-home)/graphics/oglrap ; }
oglrap-idir(){ echo $(local-base)/env/graphics/oglrap ; }
oglrap-bdir(){ echo $(oglrap-idir).build ; }

oglrap-bindir(){ echo $(oglrap-idir)/bin ; }

oglrap-scd(){  cd $(oglrap-sdir); }
oglrap-cd(){   cd $(oglrap-sdir); }

oglrap-icd(){  cd $(oglrap-idir); }
oglrap-bcd(){  cd $(oglrap-bdir); }
oglrap-name(){ echo OGLRap ; }

oglrap-wipe(){
   local bdir=$(oglrap-bdir)
   rm -rf $bdir
}


oglrap-cmake(){
   local iwd=$PWD

   local bdir=$(oglrap-bdir)
   mkdir -p $bdir
  
   oglrap-bcd 
   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(oglrap-idir) \
       $(oglrap-sdir)

   cd $iwd
}

oglrap-make(){
   local iwd=$PWD

   oglrap-bcd 
   make $*

   cd $iwd
}

oglrap-install(){
   oglrap-make install
}



oglrap--()
{
    oglrap-wipe
    oglrap-cmake
    oglrap-make
    oglrap-install

}


oglrap-export()
{
   export SHADER_DIR=$(oglrap-sdir)/glsl
} 

oglrap-run(){ 
   local bin=$(oglrap-bindir)/OGLRapTest
   oglrap-export
   $bin $*
}

oglrap-frametest()
{
   local path=${1:-/tmp/teapot.ppm}
   shift
   local bin=$(oglrap-bindir)/FrameTest
   oglrap-export
   $bin $path $*
}


