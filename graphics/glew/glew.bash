# === func-gen- : graphics/glew/glew fgp graphics/glew/glew.bash fgn glew fgh graphics/glew
glew-src(){      echo graphics/glew/glew.bash ; }
glew-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(glew-src)} ; }
glew-vi(){       vi $(glew-source) ; }
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
glew-env(){      elocal- ; opticks- ;  }

glew-fold(){ echo $(opticks-prefix)/externals/glew ; }
glew-dir(){  echo $(glew-fold)/$(glew-name) ; }
#glew-idir(){ echo $(glew-fold)/$(glew-version) ; }
glew-idir(){ echo $(glew-fold)/glew ; }
glew-prefix(){ echo $(opticks-prefix)/externals ; }


glew-sdir(){ echo $(glew-dir) ; }
glew-bdir(){ echo $(glew-dir).build ; }
glew-edir(){ echo $(opticks-home)/graphics/glew ; }

glew-cd(){   cd $(glew-sdir); }
glew-scd(){  cd $(glew-sdir); }
glew-bcd(){  cd $(glew-bdir); }
glew-icd(){  cd $(glew-idir); }
glew-ecd(){  cd $(glew-edir); }

#glew-version(){ echo 1.12.0 ; }
glew-version(){ echo 1.13.0 ; }
glew-name(){ echo glew-$(glew-version) ;}

glew-url(){ 
   local gen=$(opticks-cmake-generator)
   case $gen in
      "Visual Studio 14 2015") echo http://downloads.sourceforge.net/project/glew/glew/$(glew-version)/$(glew-name)-win32.zip ;;
                            *) echo http://downloads.sourceforge.net/project/glew/glew/$(glew-version)/$(glew-name).zip ;;
   esac
}


glew-doc(){
  local doc=$(glew-dir)/doc/index.html
  env-open $doc
}

glew-get(){
   local dir=$(dirname $(glew-dir)) &&  mkdir -p $dir && cd $dir

   local url=$(glew-url)
   local zip=$(basename $url)
   local nam=${zip/.zip}

   [ ! -f "$zip" ] && curl -L -O $url
   [ ! -d "$nam" ] && unzip $zip 

   # ln -sfnv $nam glew
}

glew-export (){ 
    export GLEW_PREFIX=$(glew-prefix)
}


glew-edit(){ vi $(opticks-home)/cmake/Modules/FindGLEW.cmake ; }


glew-make(){

   local target=${1:-install}
   local iwd=$PWD
   glew-scd

   local gen=$(opticks-cmake-generator)
   case $gen in 
      "Visual Studio 14 2015") glew-install-win ;; 
                            *) make $target GLEW_PREFIX=$(glew-prefix) GLEW_DEST=$(glew-prefix)  ;;
   esac
   cd $iwd
}


glew-install-win(){
   local msg=" === $FUNCNAME :"
   local iwd=$PWD
   glew-scd

   local bin=$(glew-prefix)/bin
   local lib=$(glew-prefix)/lib
   local inc=$(glew-prefix)/include/GL

   [ ! -d "$bin" ] && echo $msg making bin $bin && mkdir -p $bin
   [ ! -d "$lib" ] && echo $msg making lib $lib && mkdir -p $lib
   [ ! -d "$inc" ] && echo $msg making inc $inc && mkdir -p $inc

   cp -v bin/Release/Win32/glew32.dll $bin/
   cp -v lib/Release/Win32/glew32.lib $lib/
   
   local hdr
   for hdr in $(ls -1 include/GL); do
      cp -v include/GL/$hdr $inc/  
   done

   cd $iwd
}



glew--() {
    glew-get
    glew-make install
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


