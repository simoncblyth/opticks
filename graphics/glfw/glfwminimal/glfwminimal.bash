# === func-gen- : graphics/thrust_opengl_interop/glfwminimal fgp graphics/thrust_opengl_interop/glfwminimal.bash fgn glfwminimal fgh graphics/thrust_opengl_interop
glfwminimal-src(){      echo graphics/thrust_opengl_interop/glfwminimal.bash ; }
glfwminimal-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(glfwminimal-src)} ; }
glfwminimal-vi(){       vi $(glfwminimal-source) ; }
glfwminimal-usage(){ cat << EOU




EOU
}

glfwminimal-env(){      olocal- ; }

glfwminimal-sdir(){ echo $(opticks-home)/graphics/glfw/glfwminimal ; }
glfwminimal-idir(){ echo $(local-base)/env/graphics/glfw/glfwminimal ; }
glfwminimal-bdir(){ echo $(glfwminimal-idir).build ; }
glfwminimal-bindir(){ echo $(glfwminimal-idir)/bin ; }

glfwminimal-scd(){  cd $(glfwminimal-sdir); }
glfwminimal-cd(){   cd $(glfwminimal-sdir); }

glfwminimal-icd(){  cd $(glfwminimal-idir); }
glfwminimal-bcd(){  cd $(glfwminimal-bdir); }
glfwminimal-name(){ echo GLFWMinimal ; }

glfwminimal-wipe(){
   local bdir=$(glfwminimal-bdir)
   rm -rf $bdir
}

glfwminimal-cmake(){
   local iwd=$PWD

   local bdir=$(glfwminimal-bdir)
   mkdir -p $bdir
  
   glfwminimal-bcd 
   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(glfwminimal-idir) \
       $(glfwminimal-sdir)

   cd $iwd
}


glfwminimal-make(){
   local iwd=$PWD

   glfwminimal-bcd
   make $*
   cd $iwd
}

glfwminimal-install(){
   glfwminimal-make install
}

glfwminimal--()
{
    glfwminimal-wipe
    glfwminimal-cmake
    glfwminimal-make
    glfwminimal-install

}

