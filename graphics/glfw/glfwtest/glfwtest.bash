# === func-gen- : graphics/glfw/glfwtest/glfwtest fgp graphics/glfw/glfwtest/glfwtest.bash fgn glfwtest fgh graphics/glfw/glfwtest
glfwtest-src(){      echo graphics/glfw/glfwtest/glfwtest.bash ; }
glfwtest-source(){   echo ${BASH_SOURCE:-$(env-home)/$(glfwtest-src)} ; }
glfwtest-vi(){       vi $(glfwtest-source) ; }
glfwtest-env(){      elocal- ; }
glfwtest-usage(){ cat << EOU

Building GLFW using app with cmake
=====================================

* http://antongerdelan.net/opengl/hellotriangle.html

::

    delta:glfwtest blyth$ glfwtest-run
    Renderer: NVIDIA GeForce GT 750M OpenGL Engine
    OpenGL version supported 4.1 NVIDIA-8.26.26 310.40.45f01


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
glfwtest-run(){ 
   local bin=$(glfwtest-bin)
   $bin $*
}




