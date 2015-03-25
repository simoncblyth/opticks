# === func-gen- : graphics/glew/glew fgp graphics/glew/glew.bash fgn glew fgh graphics/glew
glew-src(){      echo graphics/glew/glew.bash ; }
glew-source(){   echo ${BASH_SOURCE:-$(env-home)/$(glew-src)} ; }
glew-vi(){       vi $(glew-source) ; }
glew-env(){      elocal- ; }
glew-usage(){ cat << EOU

GLEW : The OpenGL Extension Wrangler Library
==============================================

* http://glew.sourceforge.net

The OpenGL Extension Wrangler Library (GLEW) is a cross-platform open-source
C/C++ extension loading library. GLEW provides efficient run-time mechanisms
for determining which OpenGL extensions are supported on the target platform.
OpenGL core and extension functionality is exposed in a single header file.
GLEW has been tested on a variety of operating systems, including Windows,
Linux, Mac OS X, FreeBSD, Irix, and Solaris.



glew-cmake fails
-----------------

::

    -- Found PkgConfig: /opt/local/bin/pkg-config (found version "0.28") 
    -- checking for module 'gl'
    --   package 'gl' not found
    CMake Error at /opt/local/share/cmake-2.8/Modules/FindPkgConfig.cmake:279 (message):
      A required package was not found
    Call Stack (most recent call first):
      /opt/local/share/cmake-2.8/Modules/FindPkgConfig.cmake:333 (_pkg_check_modules_internal)
      CMakeLists.txt:26 (pkg_check_modules)





EOU
}
glew-dir(){  echo $(local-base)/env/graphics/glew/$(glew-name) ; }
glew-idir(){ echo $(local-base)/env/graphics/glew/$(glew-version) ; }
glew-sdir(){ echo $(glew-dir) ; }
glew-bdir(){ echo $(glew-dir).build ; }

glew-scd(){  cd $(glew-sdir); }
glew-bcd(){  cd $(glew-bdir); }
glew-icd(){  cd $(glew-idir); }

glew-version(){ echo 1.12.0 ; }
glew-name(){ echo glew-$(glew-version) ;}
glew-url(){ echo http://downloads.sourceforge.net/project/glew/glew/$(glew-version)/$(glew-name).zip ; }

glew-get(){
   local dir=$(dirname $(glew-dir)) &&  mkdir -p $dir && cd $dir

   local url=$(glew-url)
   local zip=$(basename $url)
   local nam=${zip/.zip}

   [ ! -f "$zip" ] && curl -L -O $url
   [ ! -d "$nam" ] && unzip $zip

}

glew-make(){
   local iwd=$PWD
   glew-scd
   make GLEW_PREFIX=$(glew-idir) GLEW_DEST=$(glew-idir)  
   cd $iwd
}
glew-install(){
   local iwd=$PWD
   glew-scd
   make install GLEW_PREFIX=$(glew-idir) GLEW_DEST=$(glew-idir)  
   cd $iwd
}



glew-cmake-not-working(){
   local iwd=$PWD
   local bdir=$(glew-bdir)
   mkdir -p $bdir
   glew-bcd
   cmake $(glew-dir)
   cd $iwd
}
glew-make-not-working(){
   local iwd=$PWD
   glew-bcd
   make $*
   cd $iwd
}


