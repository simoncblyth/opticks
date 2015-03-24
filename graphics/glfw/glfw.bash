# === func-gen- : graphics/glfw/glfw fgp graphics/glfw/glfw.bash fgn glfw fgh graphics/glfw
glfw-src(){      echo graphics/glfw/glfw.bash ; }
glfw-source(){   echo ${BASH_SOURCE:-$(env-home)/$(glfw-src)} ; }
glfw-vi(){       vi $(glfw-source) ; }
glfw-env(){      elocal- ; }
glfw-usage(){ cat << EOU

GLFW
======

* http://www.glfw.org

GLFW is an Open Source, multi-platform library for creating windows with OpenGL
contexts and receiving input and events. It is easy to integrate into existing
applications and does not lay claim to the main loop.

Version 3.1.1 released on March 19, 2015




EOU
}
glfw-sdir(){ echo $(env-home)/graphics/glfw ; }
glfw-dir(){  echo $(local-base)/env/graphics/glfw/$(glfw-name) ; }
glfw-bdir(){ echo $(glfw-dir).build ; }
glfw-idir(){ echo $(local-base)/env/graphics/glfw/$(glfw-version) ; }

glfw-scd(){ cd $(glfw-sdir); }
glfw-cd(){  cd $(glfw-dir); }
glfw-bcd(){ cd $(glfw-bdir); }
glfw-icd(){ cd $(glfw-idir); }

glfw-version(){ echo 3.1.1 ; }
glfw-name(){ echo glfw-$(glfw-version) ; }
glfw-url(){  echo http://downloads.sourceforge.net/project/glfw/glfw/$(glfw-version)/$(glfw-name).zip ; }

glfw-get(){
   local dir=$(dirname $(glfw-dir)) &&  mkdir -p $dir && cd $dir

   local url=$(glfw-url)
   local zip=$(basename $url)
   local nam=${zip/.zip}
   [ ! -f "$zip" ] && curl -L -O $url
   [ ! -d "$nam" ] && unzip $zip 

}


glfw-wipe(){
  local bdir=$(glfw-bdir)
  rm -rf $bdir
}

glfw-cmake(){
  local iwd=$PWD

  local bdir=$(glfw-bdir)
  mkdir -p $bdir

  glfw-bcd
  cmake -DCMAKE_INSTALL_PREFIX=$(glfw-idir) $(glfw-dir)

  cd $iwd
}

glfw-make(){
  local iwd=$PWD

  glfw-bcd
  make $* 

  cd $iwd
}


