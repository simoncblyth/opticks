root-source(){   echo $BASH_SOURCE ; }
root-vi(){       vi $(root-source) ; }
root-env(){      echo -n ; }
root-usage(){ cat << EOU

* https://root.cern/install/
* https://root.cern/releases/release-62206/


root-cmake fatal::

    -- Found GLEW: /opt/local/include (found version "2.1.0") 
    CMake Error at cmake/modules/SearchInstalledSoftware.cmake:538 (message):
      Please enable builtin Glew due bug in latest CMake (use cmake option
      -Dbuiltin_glew=ON).
    Call Stack (most recent call first):
      CMakeLists.txt:192 (include)


/usr/local/opticks_externals/root_6.22.06.build/root-6.22.06/cmake/modules/SearchInstalledSoftware.cmake::

     526 #---Check for GLEW -------------------------------------------------------------------
     527 # Opengl is "must" requirement for Glew.
     528 if(opengl AND NOT builtin_glew)
     529   message(STATUS "Looking for GLEW")
     530   if(fail-on-missing)
     531     find_package(GLEW REQUIRED)
     532   else()
     533     find_package(GLEW)
     534     # Bug was reported on newer version of CMake on Mac OS X:
     535     # https://gitlab.kitware.com/cmake/cmake/-/issues/19662
     536     # https://github.com/microsoft/vcpkg/pull/7967
     537     if(GLEW_FOUND AND APPLE AND CMAKE_VERSION VERSION_GREATER 3.15)
     538       message(FATAL_ERROR "Please enable builtin Glew due bug in latest CMake (use cmake option -Dbuiltin_glew=ON).")
     539       unset(GLEW_FOUND)
     540     elseif(GLEW_FOUND AND NOT TARGET GLEW::GLEW)
     541       add_library(GLEW::GLEW UNKNOWN IMPORTED)
     542       set_target_properties(GLEW::GLEW PROPERTIES
     543       IMPORTED_LOCATION "${GLEW_LIBRARIES}"
     544       INTERFACE_INCLUDE_DIRECTORIES "${GLEW_INCLUDE_DIRS}")
     545     endif()
     546     if(NOT GLEW_FOUND)
     547       message(STATUS "GLEW not found. Switching on builtin_glew option")
     548       set(builtin_glew ON CACHE BOOL "Enabled because opengl requested and GLEW not found (${builtin_glew_description})" FORCE)
     549     endif()
     550   endif()
     551 endif()


[ 46%] Built target Dictgen
Scanning dependencies of target rootcling_stage1
[ 46%] Building CXX object core/rootcling_stage1/CMakeFiles/rootcling_stage1.dir/src/rootcling_stage1.cxx.o
[ 46%] Linking CXX executable src/rootcling_stage1
[ 46%] Built target rootcling_stage1
Scanning dependencies of target Macosx
[ 47%] Building CXX object core/macosx/CMakeFiles/Macosx.dir/src/CocoaUtils.mm.o
[ 47%] Building CXX object core/macosx/CMakeFiles/Macosx.dir/src/TMacOSXSystem.mm.o
[ 47%] Built target Macosx
Scanning dependencies of target rconfigure
[ 47%] Generating ../include/RConfigure.h
[ 47%] Built target rconfigure
[ 47%] Generating G__Core.cxx, ../lib/Core.pcm
While building module 'Core':
While building module 'std' imported from input_line_1:1:
While building module 'Darwin' imported from /Applications/Xcode/Xcode_9_2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include/c++/v1/ctype.h:39:
In file included from <module-includes>:1064:
In file included from /usr/include/ncurses.h:141:
/opt/local/include/unctrl.h:61:38: error: cannot initialize a variable of type 'char *' with an lvalue of type 'char *(chtype)' (aka 'char *(unsigned int)')
NCURSES_EXPORT(NCURSES_CONST char *) NCURSES_SP_NAME(unctrl) (SCREEN*, chtype);
                                     ^               ~~~~~~
/opt/local/include/unctrl.h:61:61: error: expected ';' after top level declarator
NCURSES_EXPORT(NCURSES_CONST char *) NCURSES_SP_NAME(unctrl) (SCREEN*, chtype);
                                                            ^
While building module 'Core':
While building module 'std' imported from input_line_1:1:
In file included from <module-includes>:2:
/Applications/Xcode/Xcode_9_2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include/c++/v1/ctype.h:39:15: fatal error: could not build module 'Darwin'
#include_next <ctype.h>
 ~~~~~~~~~~~~~^
input_line_1:1:10: fatal error: could not build module 'std'
#include <new>
 ~~~~~~~~^
Warning in cling::IncrementalParser::CheckABICompatibility():
  Failed to extract C++ standard library version.
Warning in cling::IncrementalParser::CheckABICompatibility():
  Possible C++ standard library mismatch, compiled with _LIBCPP_ABI_VERSION '1'
  Extraction of runtime standard library version was: ''
While building module 'Core':
While building module 'Cling_Runtime' imported from input_line_2:1:
In file included from <module-includes>:1:
/usr/local/opticks_externals/root_6.22.06.build/root-6.22.06.Debug.build/etc/cling/Interpreter/RuntimeUniverse.h:26:10: fatal error: could not build module 'std'
#include <new>
 ~~~~~~~~^
/usr/local/opticks_externals/root_6.22.06.build/root-6.22.06.Debug.build/etc/cling/Interpreter/DynamicExprInfo.h:13:10: fatal error: could not build module 'std'
#include <string>
 ~~~~~~~~^
While building module 'Core':
While building module '_Builtin_intrinsics':
In file included from <module-includes>:2:
In file included from /usr/local/opticks_externals/root_6.22.06.build/root-6.22.06.Debug.build/etc/cling/lib/clang/5.0.0/include/immintrin.h:32:
In file included from /usr/local/opticks_externals/root_6.22.06.build/root-6.22.06.Debug.build/etc/cling/lib/clang/5.0.0/include/xmmintrin.h:39:
In file included from /usr/local/opticks_externals/root_6.22.06.build/root-6.22.06.Debug.build/etc/cling/lib/clang/5.0.0/include/mm_malloc.h:27:
In file included from /Applications/Xcode/Xcode_9_2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include/c++/v1/stdlib.h:94:
/usr/local/opticks_externals/root_6.22.06.build/root-6.22.06.Debug.build/etc/cling/lib/clang/5.0.0/include/stdlib.h:8:15: fatal error: could not build module 'Darwin'
#include_next <stdlib.h>
 ~~~~~~~~~~~~~^
While building module 'Core':
<<< cling interactive line includer >>>: fatal error: could not build module '_Builtin_intrinsics'
Failed to load module _Builtin_intrinsics
Error: Error loading the default header files.
make[2]: *** [core/G__Core.cxx] Error 1
make[1]: *** [core/CMakeFiles/G__Core.dir/all] Error 2
make: *** [all] Error 2
=== root-build : ABORT
epsilon:root-6.22.06.Debug.build blyth$ 


EOU
}

root-ver(){ echo 6.22.06 ; }
root-name(){ echo root-$(root-ver) ; }  # name of exploded distribution dir
root-url(){ echo https://root.cern/download/root_v$(root-ver).source.tar.gz ; }

root-prefix(){          echo $(root-prefix-default) ; }
root-prefix-default(){  echo $(opticks-prefix)_externals/root_$(root-ver)  ; }
root-prefix-frompath(){ echo opticks-setup-find-config-prefix ROOT  ; }

root-sdir(){             echo $(root-prefix).build/$(root-name) ; }  # exploded distribution dir
root-scd(){              cd $(root-sdir)/$1 ; }
root-cd(){               cd $(root-sdir)/$1 ; }

root-buildtype(){       echo Debug ; }
root-bdir(){            echo $(root-sdir).$(root-buildtype).build ; }
root-bcd(){             local bdir=$(root-bdir) ; mkdir -p $bdir ; cd $bdir/$1 ; }

root-get(){
   local dir=$(dirname $(root-dir)) &&  mkdir -p $dir && cd $dir
   local url=$(root-url)
   local dst=$(basename $url)
   local nam=$(root-name)

   [ ! -f "$dst" ] && echo getting $url && curl -L -O $url
   [ ! -d "$nam" ] && echo exploding tarball && tar zxvf $dst 
   [ -d "$nam" ]
}


root--(){
   root-get
}


root-cmake-notes(){ cat << EON

* https://root.cern/install/build_from_source/

Without fail-on-missing=ON there is a fatal cmake error
suggesting should use builtin_glew.  But cannot 
do that because Opticks depends on glew. 


EON
}

root-cmake(){
   local msg="=== $FUNCNAME :"
   root-bcd
   cmake \
      -DCMAKE_BUILD_TYPE=$(root-buildtype) \
      -DCMAKE_INSTALL_PREFIX=$(root-prefix) \
      -DCMAKE_CXX_STANDARD=14 \
      -Dmysql=OFF \
      -Doracle=OFF \
      -Dpgsql=OFF \
      -Dpythia6=OFF \
      -Dpythia8=OFF \
      -Dxrootd=OFF \
      -Dgfal=OFF \
      -Ddavix=OFF \
      -Dimt=OFF \
      -Dvdt=OFF \
      -Dcuda=OFF \
      -Dcudnn=OFF \
      -Dtmva=OFF \
      -Dtmva-gpu=OFF \
      -Dbuiltin_ftgl=ON \
      -Dbuiltin_cfitsio=ON \
      -Dfail-on-missing=ON \
       $(root-sdir)

   local rc=$?
   [ $rc -ne 0 ] && echo $msg RC $rc 
   return $rc
}
root-configure-msg(){ cat << EOM
Already configured, use root-cmake to do it again   
EOM
}

root-configure()
{
   local bdir=$(root-bdir)
   [ -f "$bdir/CMakeCache.txt" ] && root-configure-msg  && return
   root-cmake $*
   return $?
}

root-build(){
   local msg="=== $FUNCNAME :"
   root-configure
   [ $? -ne 0 ] && echo $msg ABORT && return 

   root-bcd
   make
   [ $? -ne 0 ] && echo $msg ABORT && return 
   make install 
   [ $? -ne 0 ] && echo $msg ABORT && return 
}
