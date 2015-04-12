# === func-gen- : graphics/glfw/glfwtest/glfwtest fgp graphics/glfw/glfwtest/glfwtest.bash fgn glfwtest fgh graphics/glfw/glfwtest
glfwtest-src(){      echo graphics/glfw/glfwtest/glfwtest.bash ; }
glfwtest-source(){   echo ${BASH_SOURCE:-$(env-home)/$(glfwtest-src)} ; }
glfwtest-vi(){       vi $(glfwtest-source) ; }
glfwtest-usage(){ cat << EOU

Building GLFW using app with cmake
=====================================

* http://antongerdelan.net/opengl/hellotriangle.html

With hinting get OpenGL 4.1::

    delta:glfwtest blyth$ glfwtest-run
    Renderer: NVIDIA GeForce GT 750M OpenGL Engine
    OpenGL version supported 4.1 NVIDIA-8.26.26 310.40.45f01

Without hinting get OpenGL 2.1::

    Renderer: NVIDIA GeForce GT 750M OpenGL Engine
    OpenGL version supported 2.1 NVIDIA-8.26.26 310.40.45f01



Find GLFW without using FindGLFW.cmake
----------------------------------------

Via pkg-config mechanism::

    delta:glfwtest blyth$ PKG_CONFIG_PATH=$(glfw-idir)/lib/pkgconfig glfwtest-cmake
    GLFW_INCLUDE_DIR:/usr/local/env/graphics/glfw/3.1.1/include
    GLFW_LIBRARY:/usr/local/env/graphics/glfw/3.1.1/lib/libglfw3.a
    GLFW_DEFINITIONS:
    -- Configuring done


http://www.cmake.org/Wiki/CMake:How_To_Find_Libraries


EOU
}
glfwtest-sdir(){ echo $(env-home)/graphics/glfw/glfwtest ; }
glfwtest-idir(){ echo $(local-base)/env/graphics/glfw/glfwtest ; }
glfwtest-bdir(){ echo $(glfwtest-idir).build ; }

glfwtest-scd(){  cd $(glfwtest-sdir); }
glfwtest-cd(){  cd $(glfwtest-sdir); }

glfwtest-icd(){  cd $(glfwtest-idir); }
glfwtest-bcd(){  cd $(glfwtest-bdir); }
glfwtest-name(){ echo GLFWTest ; }

glfwtest-env(){      
   elocal- 
   optix-
   optix-export 

}

glfwtest-wipe(){
   local bdir=$(glfwtest-bdir)
   rm -rf $bdir
}

glfwtest-cmake-pc(){
   glfw-
   PKG_CONFIG_PATH=$(glfw-idir)/lib/pkgconfig glfwtest-cmake
}

glfwtest-cmake(){
   local iwd=$PWD

   local bdir=$(glfwtest-bdir)
   mkdir -p $bdir
  
   glfwtest-bcd 
   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(glfwtest-idir) \
       $(glfwtest-sdir)

   cd $iwd
}

glfwtest-make(){
   local iwd=$PWD

   glfwtest-bcd 
   make $*

   cd $iwd
}

glfwtest-install(){
   glfwtest-make install
}

glfwtest-bin(){ echo $(glfwtest-idir)/bin/$(glfwtest-name) ; }
glfwtest-export()
{
   export-
   export-export

   export GLFWTEST_GEOKEY="DAE_NAME_DYB"
   export GLFWTEST_QUERY="range:5000:8000"
   export GLFWTEST_CTRL=""
   export SHADER_DIR=$(glfwtest-sdir)/gl
} 
glfwtest-run(){ 
   local bin=$(glfwtest-bin)
   glfwtest-export
   $bin $* 
}

glfwtest-runq(){
   local bin=$(glfwtest-bin)
   glfwtest-export

   ## bash drops quotes, so put them back when the parameter
   ## contains a space
   ##
   ## for example the liveline parameter quotes must be preserved
   ##  
   ## /usr/local/env/graphics/glfw/glfwtest/bin/GLFWTest --version --yfov 123 --yfov 456 --config demo.cfg --liveline "--yfov 1234"

   local parms="" 
   local p
   for p in "$@" ; do
      [ "${p/ /}" == "$p" ] && parms="${parms} $p" || parms="${parms} \"${p}\""
   done

   cat << EOC  | sh 
   $bin $parms
EOC
}


glfwtest--()
{
    glfwtest-wipe
    glfwtest-cmake
    glfwtest-make
    glfwtest-install
}


glfwtest-lldb()
{
   glfwtest-export
   lldb $(glfwtest-bin) $*
}

