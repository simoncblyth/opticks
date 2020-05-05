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

oimplicitmesher-src(){      echo externals/oimplicitmesher.bash ; }
oimplicitmesher-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(oimplicitmesher-src)} ; }
oimplicitmesher-vi(){       vi $(oimplicitmesher-source) ; }
oimplicitmesher-env(){      olocal- ; opticks- ; }
oimplicitmesher-usage(){ cat << EOU

ImplicitMesher as Opticks External 
====================================

See also env-;implicitmesher-

NB uses same prefix as Opticks so that opticks/cmake/Modules/FindGLM.cmake succeeds
this has knock effect of requiring prefixing in the CMake install locations::

    install(TARGETS ${name}  DESTINATION externals/lib)
    install(FILES ${HEADERS} DESTINATION externals/include/${name})

issue : two libs, and picking old one
----------------------------------------------

simon:opticks blyth$ l /usr/local/opticks/lib/libNPY.dylib 
-rwxr-xr-x  1 blyth  staff  5951412 Jun 15 19:01 /usr/local/opticks/lib/libNPY.dylib

simon:opticks blyth$ otool -L /usr/local/opticks/lib/libNPY.dylib 
/usr/local/opticks/lib/libNPY.dylib:
    @rpath/libNPY.dylib (compatibility version 0.0.0, current version 0.0.0)
    /opt/local/lib/libboost_system-mt.dylib (compatibility version 0.0.0, current version 0.0.0)
    /opt/local/lib/libboost_program_options-mt.dylib (compatibility version 0.0.0, current version 0.0.0)
    /opt/local/lib/libboost_filesystem-mt.dylib (compatibility version 0.0.0, current version 0.0.0)
    /opt/local/lib/libboost_regex-mt.dylib (compatibility version 0.0.0, current version 0.0.0)
    @rpath/libSysRap.dylib (compatibility version 0.0.0, current version 0.0.0)
    @rpath/libBoostRap.dylib (compatibility version 0.0.0, current version 0.0.0)
    @rpath/libOpenMeshCore.6.3.dylib (compatibility version 6.3.0, current version 6.3.0)
    @rpath/libOpenMeshTools.6.3.dylib (compatibility version 6.3.0, current version 6.3.0)
    /usr/lib/libssl.0.9.8.dylib (compatibility version 0.9.8, current version 50.0.0)
    /usr/lib/libcrypto.0.9.8.dylib (compatibility version 0.9.8, current version 50.0.0)
    @rpath/libImplicitMesher.dylib (compatibility version 0.0.0, current version 0.0.0)
    @rpath/libDualContouringSample.dylib (compatibility version 0.0.0, current version 0.0.0)
    @rpath/libYoctoGL.dylib (compatibility version 0.0.0, current version 0.0.0)
    /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 120.0.0)
    /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1197.1.1)
simon:opticks blyth$ 
simon:opticks blyth$ 
simon:opticks blyth$ otool -L /usr/local/opticks/lib/libImplicitMesher.dylib 
/usr/local/opticks/lib/libImplicitMesher.dylib:
    @rpath/libImplicitMesher.dylib (compatibility version 0.0.0, current version 0.0.0)
    /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 120.0.0)
    /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1197.1.1)
simon:opticks blyth$ 
simon:opticks blyth$ ll /usr/local/opticks/lib/libImplicitMesher.dylib 
-rwxr-xr-x  1 blyth  staff  157728 Jun 14 13:17 /usr/local/opticks/lib/libImplicitMesher.dylib
simon:opticks blyth$ 
simon:opticks blyth$ ll /usr/local/opticks/externals/lib/libImplicitMesher.dylib 
-rwxr-xr-x  1 blyth  staff  157728 Jun 15 18:55 /usr/local/opticks/externals/lib/libImplicitMesher.dylib
simon:opticks blyth$ 

::

    simon:lib blyth$ pwd
    /usr/local/opticks/lib
    simon:lib blyth$ l *.dylib
    -rwxr-xr-x  1 blyth  staff  5951412 Jun 15 19:01 libNPY.dylib
    -rwxr-xr-x  1 blyth  staff   144052 Jun 15 14:55 libokg4.dylib
    -rwxr-xr-x  1 blyth  staff  1909184 Jun 15 14:55 libcfg4.dylib
    -rwxr-xr-x  1 blyth  staff   119116 Jun 15 14:55 libOK.dylib
    -rwxr-xr-x  1 blyth  staff   213984 Jun 15 14:55 libOpticksGL.dylib
    -rwxr-xr-x  1 blyth  staff   798016 Jun 15 14:55 libOKOP.dylib
    -rwxr-xr-x  1 blyth  staff  2145020 Jun 15 14:55 libOptiXRap.dylib
    -rwxr-xr-x  1 blyth  staff  3764296 Jun 15 14:54 libThrustRap.dylib
    -rwxr-xr-x  1 blyth  staff  1104416 Jun 15 14:54 libOGLRap.dylib
    -rwxr-xr-x  1 blyth  staff   228728 Jun 15 14:54 libOpticksGeometry.dylib
    -rwxr-xr-x  1 blyth  staff   846852 Jun 15 14:54 libOpenMeshRap.dylib
    -rwxr-xr-x  1 blyth  staff   280080 Jun 15 14:54 libAssimpRap.dylib
    -rwxr-xr-x  1 blyth  staff  2639276 Jun 15 14:54 libGGeo.dylib
    -rwxr-xr-x  1 blyth  staff  2339208 Jun 15 14:54 libOpticksCore.dylib
    -rwxr-xr-x  1 blyth  staff  4369952 Jun 14 17:03 libBoostRap.dylib
    -rwxr-xr-x  1 blyth  staff   616392 Jun 14 16:21 libCUDARap.dylib
    -rwxr-xr-x  1 blyth  staff   144964 Jun 14 15:30 libSysRap.dylib

    -rwxr-xr-x  1 blyth  staff   102404 Jun 14 14:54 libDualContouringSample.dylib
    -rwxr-xr-x  1 blyth  staff  4773576 Jun 14 13:50 libxerces-c-3.1.dylib
    lrwxr-xr-x  1 blyth  staff       21 Jun 14 13:50 libxerces-c.dylib -> libxerces-c-3.1.dylib
    -rwxr-xr-x  1 blyth  staff   157728 Jun 14 13:17 libImplicitMesher.dylib
    -rwxr-xr-x  1 blyth  staff  3441116 Jun 14 13:15 libYoctoGL.dylib

Get rid of imposters::

    simon:lib blyth$ rm -f libDualContouringSample.dylib libxerces-c-3.1.dylib libxerces-c.dylib libImplicitMesher.dylib libYoctoGL.dylib



Avoided the below on Darwin by sleeping 2s between make and install
---------------------------------------------------------------------

::


    error: /opt/local/bin/install_name_tool: no LC_RPATH load command with path: /usr/local/opticks/externals/ImplicitMesher/ImplicitMesher.build found in: /usr/local/opticks/lib/ImplicitMesherFTest (for architecture x86_64), required for specified option "-delete_rpath /usr/local/opticks/externals/ImplicitMesher/ImplicitMesher.build"



EOU
}

oimplicitmesher-edit(){ vi $(opticks-home)/cmake/Modules/FindImplicitMesher.cmake ; }

oimplicitmesher-url-http(){ echo https://bitbucket.com/simoncblyth/ImplicitMesher ; }
oimplicitmesher-url-ssh(){  echo ssh://hg@bitbucket.org/simoncblyth/ImplicitMesher ; }
oimplicitmesher-url(){
   case $USER in 
      blyth) oimplicitmesher-url-ssh ;;
          *) oimplicitmesher-url-http ;; 
   esac
}



oimplicitmesher-dir(){  echo $(opticks-prefix)/externals/ImplicitMesher/ImplicitMesher ; }
oimplicitmesher-bdir(){ echo $(opticks-prefix)/externals/ImplicitMesher/ImplicitMesher.build ; }



oimplicitmesher-cd(){  cd $(oimplicitmesher-dir); }
oimplicitmesher-bcd(){ cd $(oimplicitmesher-bdir) ; }


oimplicitmesher-fullwipe()
{
    local iwd=$PWD
    cd $(opticks-prefix)

    rm -rf externals/ImplicitMesher/ImplicitMesher.build

    rm -f  externals/lib/libImplicitMesher.*
    rm -rf externals/include/ImplicitMesher
    rm -rf externals/lib/cmake/implicitmesher
    rm -f  externals/lib/pkgconfig/implicitmesher.pc

    rm -rf include/ImplicitMesher

    rm -f  lib/libImplicitMesher.*
    rm -rf lib/cmake/implicitmesher
    rm -f  lib/pkgconfig/implicitmesher.pc

    rm -f  lib64/libImplicitMesher.*
    rm -rf lib64/cmake/implicitmesher
    rm -f  lib64/pkgconfig/implicitmesher.pc


    cd $iwd
}

oimplicitmesher-update()
{
    oimplicitmesher-fullwipe
    oimplicitmesher-- 
}

oimplicitmesher-info(){ cat << EOI

    oimplicitmesher-url  : $(oimplicitmesher-url)
    oimplicitmesher-dir  : $(oimplicitmesher-dir)

EOI
}


oimplicitmesher-get(){
   local iwd=$PWD
   local dir=$(dirname $(oimplicitmesher-dir)) &&  mkdir -p $dir && cd $dir
   local url=$(oimplicitmesher-url)
   local nam=$(basename $url)
   [ ! -d $nam ] && hg clone $url
   cd $iwd
}

oimplicitmesher-cmake()
{
    local iwd=$PWD
    local rc
    local bdir=$(oimplicitmesher-bdir)

    mkdir -p $bdir
    #[ -f "$bdir/CMakeCache.txt" ] && echo $msg configured already && return  
    rm -f "$bdir/CMakeCache.txt"

    oimplicitmesher-bcd   
    opticks-

    cmake \
       -DOPTICKS_PREFIX=$(opticks-prefix) \
       -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules \
       -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
       -DCMAKE_BUILD_TYPE=$(opticks-buildtype) \
       -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
      $* \
       $(oimplicitmesher-dir)

    rc=$?
    cd $iwd
    return $rc
}

oimplicitmesher-cmake-notes(){ cat << EON

OpticksBuildOptions.cmake sets CMAKE_INSTALL_INCLUDEDIR to include/$name 
where name is ImplicitMesher. Because of this cannot control from the 
cmake commandline::

       -DCMAKE_INSTALL_LIBDIR=externals/lib \
       -DCMAKE_INSTALL_INCLUDEDIR=externals/include/ImplicitMesher \
 
Instead have to do it within the CMakeLists.txt


EON
}


oimplicitmesher-make()
{
    local iwd=$PWD
    local rc
    oimplicitmesher-bcd

    #cmake --build . --config $(opticks-buildtype) --target ${1:-install}
    make $1

    rc=$?
    cd $iwd
    return $rc
}

oimplicitmesher-pc(){ echo $FUNCNAME placeholder ; }

oimplicitmesher--()
{
   local msg="=== $FUNCNAME :"
   oimplicitmesher-get
   [ $? -ne 0 ] && echo $msg get FAIL && return 1
   oimplicitmesher-cmake
   [ $? -ne 0 ] && echo $msg cmake FAIL && return 2
   oimplicitmesher-make
   [ $? -ne 0 ] && echo $msg build FAIL && return 3

   #[ $(uname) == "Darwin" ] && sleep 2 && echo $msg after sleep 

   oimplicitmesher-make install
   [ $? -ne 0 ] && echo $msg install FAIL && return 4
   oimplicitmesher-pc
   [ $? -ne 0 ] && echo $msg pc FAIL && return 5
   return 0
}

oimplicitmesher-t()
{
   oimplicitmesher-make test
}


oimplicitmesher-setup(){ cat << EOS
# $FUNCNAME
EOS
}

