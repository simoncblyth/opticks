# === func-gen- : graphics/thrust_opengl_interop/glewminimal fgp graphics/thrust_opengl_interop/glewminimal.bash fgn glewminimal fgh graphics/thrust_opengl_interop
glewminimal-src(){      echo graphics/thrust_opengl_interop/glewminimal.bash ; }
glewminimal-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(glewminimal-src)} ; }
glewminimal-vi(){       vi $(glewminimal-source) ; }
glewminimal-usage(){ cat << EOU




EOU
}

glewminimal-env(){      elocal- ; }

glewminimal-sdir(){ echo $(opticks-home)/graphics/glew/glewminimal ; }
glewminimal-idir(){ echo $(local-base)/env/graphics/glew/glewminimal ; }
glewminimal-bdir(){ echo $(glewminimal-idir).build ; }
glewminimal-bindir(){ echo $(glewminimal-idir)/bin ; }

glewminimal-scd(){  cd $(glewminimal-sdir); }
glewminimal-cd(){   cd $(glewminimal-sdir); }

glewminimal-icd(){  cd $(glewminimal-idir); }
glewminimal-bcd(){  cd $(glewminimal-bdir); }
glewminimal-name(){ echo GLFWMinimal ; }

glewminimal-wipe(){
   local bdir=$(glewminimal-bdir)
   rm -rf $bdir
}

glewminimal-cmake(){
   local iwd=$PWD

   local bdir=$(glewminimal-bdir)
   mkdir -p $bdir
  
   glewminimal-bcd 
   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(glewminimal-idir) \
       $(glewminimal-sdir)

   cd $iwd
}


glewminimal-make(){
   local iwd=$PWD

   glewminimal-bcd
   make $*
   cd $iwd
}

glewminimal-install(){
   glewminimal-make install
}

glewminimal--()
{
    glewminimal-wipe
    glewminimal-cmake
    glewminimal-make
    glewminimal-install

}

