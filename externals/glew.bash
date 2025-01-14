##
## Copyright (c) 2019 Opticks Team. All Rights Reserved.
##
## This file is part of Opticks
## (see https://bitbucket.org/simoncblyth/opticks).
##
## Licensed under the Apache License, Version 2.0 (the "License"); 
## you may not use this file except in compliance with the License.  
## You may obtain a copy of the License at
##
##   http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software 
## distributed under the License is distributed on an "AS IS" BASIS, 
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
## See the License for the specific language governing permissions and 
## limitations under the License.
##

# === func-gen- : graphics/glew/glew fgp externals/glew.bash fgn glew fgh graphics/glew
glew-src(){      echo externals/glew.bash ; }
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


The ROOT distrib includes libGLEW.so that can cause issues
-------------------------------------------------------------


With glew 1.13.0 glew-cmake fails, so are using the Makefile 
----------------------------------------------------------------

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
glew-env(){      olocal- ; opticks- ;  }

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
#glew-version(){ echo 2.1.0 ; }
glew-name(){ echo glew-$(glew-version) ;}
glew-libdir(){ echo $(glew-prefix)/lib ; }


glew-info(){ cat << EOI

   glew-version : $(glew-version)
   glew-name    : $(glew-name)   
   glew-sdir    : $(glew-sdir) 
   glew-url     : $(glew-url) 
   glew-dist    : $(glew-dist) 
   glew-prefix  : $(glew-prefix) 
   glew-libdir  : $(glew-libdir) 


EOI
}


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

glew-dist(){ echo $(dirname $(glew-dir))/$(basename $(glew-url)) ; }
glew-get(){
   local dir=$(dirname $(glew-dir)) &&  mkdir -p $dir && cd $dir

   local url=$(glew-url)
   local zip=$(basename $url)
   local nam=${zip/.zip}
   local opt=$( [ -n "${VERBOSE}" ] && echo "" || echo "-q" )

   [ ! -f "$zip" ] && opticks-curl $url
   [ ! -d "$nam" ] && unzip $opt $zip 

   [ -d "$nam" ] 
}

glew-export (){ 
    export GLEW_PREFIX=$(glew-prefix)
}


glew-edit(){ vi $(opticks-home)/cmake/Modules/FindGLEW.cmake ; }



glew-make-notes(){ cat << EON

* LIBDIR override is to install into lib rather than lib64
  see notes/issues/glew-is-only-external-other-that-geant4-installing-into-lib64.rst

EON
}

glew-make(){
   local rc=0
   local target=${1:-install}
   local iwd=$PWD
   glew-scd

   local gen=$(opticks-cmake-generator)
   case $gen in 
      "Visual Studio 14 2015") glew-install-win ;; 
                            *) make $target GLEW_PREFIX=$(glew-prefix) GLEW_DEST=$(glew-prefix) LIBDIR=$(glew-prefix)/lib  ;;
   esac
   rc=$?

   cd $iwd
   return $rc
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
    local msg="=== $FUNCNAME :"   

    glew-get
    [ $? -ne 0 ] && echo $msg get FAIL && return 1
    glew-make install
    [ $? -ne 0 ] && echo $msg install FAIL && return 2
    #glew-pc
    #[ $? -ne 0 ] && echo $msg pc FAIL && return 3

    return 0 
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


glew-lib64-rm()
{
   local iwd=$PWD
   cd $(glew-prefix)/lib64
   
   ls libGLEW*
   rm libGLEW*
   rm -rf pkgconfig 
 
   cd $iwd 
}


glew-pc() 
{ 
    local msg="=== $FUNCNAME :";
    local path="$OPTICKS_PREFIX/externals/lib/pkgconfig/glew.pc";
    local path2="$OPTICKS_PREFIX/externals/lib/pkgconfig/OpticksGLEW.pc";
    if [ -f "$path2" -a ! -f "$path" ]; then
        echo $msg path2 already exists $path2 
    elif [ -f "$path" ]; then
        $(opticks-home)/bin/pc.py $path --fix;

        if [ "$(uname)" == "Darwin" ]; then 
            perl -pi -e 's/^Requires: glu/#Requires: glu/' $path 
        fi 

        mv $path $path2

    else
        echo $msg no such path $path;
    fi
}

glew-setup(){ cat << EOS
# $FUNCNAME
EOS
}


glew-manifest(){ cat << EOP
include/GL/wglew.h
include/GL/glew.h
include/GL/glxew.h
lib/libGLEW.a
lib/libGLEW.so
lib/libGLEW.so.1.13
lib/libGLEW.so.1.13.0
lib/pkgconfig/glew.pc
EOP
}

glew-manifest-wipe(){
  local pfx=$(glew-prefix)
  cd $pfx 
  [ $? -ne 0 ] && return 1

  local rel 
  echo "# $FUNCNAME "
  echo "cd $PWD" 
  for rel in $(glew-manifest) 
  do  
      if [ -d "$rel" ]; then 
          echo rm -rf \"$rel\"
      elif [ -f "$rel" ]; then 
          echo rm -f \"$rel\"
      fi    
  done
}



