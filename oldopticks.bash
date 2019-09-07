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

oldopticks(){ cat << EON

This is a resting place for old bash functions 
from opticks.bash before they get deleted.


EON
}



opticks-cmake-generator(){ echo ${OPTICKS_CMAKE_GENERATOR:-Unix Makefiles} ; }
opticks-cmake-generator-old()
{
    if [ "$NODE_TAG" == "M" ]; then
       echo MSYS Makefiles 
    else  
       case $(uname -s) in
         MINGW64_NT*)  echo Visual Studio 14 2015 ;;
                   *)  echo Unix Makefiles ;;
       esac                          
    fi
}

opticks-cmake-info(){ g4- ; xercesc- ; cat << EOI

$FUNCNAME
======================

       NODE_TAG                   :  $NODE_TAG

       opticks-sdir               :  $(opticks-sdir)
       opticks-bdir               :  $(opticks-bdir)
       opticks-cmake-generator    :  $(opticks-cmake-generator)
       opticks-compute-capability :  $(opticks-compute-capability)
       opticks-prefix             :  $(opticks-prefix)
       opticks-optix-install-dir  :  $(opticks-optix-install-dir)
       g4-cmake-dir               :  $(g4-cmake-dir)
       xercesc-library            :  $(xercesc-library)
       xercesc-include-dir        :  $(xercesc-include-dir)

EOI
}


opticks-cmakecache(){ echo $(opticks-bdir)/CMakeCache.txt ; }

opticks-cmakecache-grep(){ grep ${1:-COMPUTE_CAPABILITY} $(opticks-cmakecache) ; }
opticks-cmakecache-vars-(){  cat << EOV
CMAKE_BUILD_TYPE
COMPUTE_CAPABILITY
CMAKE_INSTALL_PREFIX
OptiX_INSTALL_DIR
Geant4_DIR
XERCESC_LIBRARY
XERCESC_INCLUDE_DIR
EOV
}

opticks-cmakecache-vars(){ 
   local var 
   $FUNCNAME- | while read var ; do
       opticks-cmakecache-grep $var 
   done    
}


opticks-cmake(){
   local msg="=== $FUNCNAME : "
   local iwd=$PWD
   local bdir=$(opticks-bdir)

   echo $msg configuring installation

   mkdir -p $bdir
   [ -f "$bdir/CMakeCache.txt" ] && echo $msg configured already use opticks-configure to wipe build dir and re-configure && return  

   opticks-bcd

   g4- 
   xercesc-

   opticks-cmake-info 

   cmake \
        -G "$(opticks-cmake-generator)" \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCOMPUTE_CAPABILITY=$(opticks-compute-capability) \
       -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
       -DOptiX_INSTALL_DIR=$(opticks-optix-install-dir) \
       -DGeant4_DIR=$(g4-cmake-dir) \
       -DXERCESC_LIBRARY=$(xercesc-library) \
       -DXERCESC_INCLUDE_DIR=$(xercesc-include-dir) \
       $* \
       $(opticks-sdir)

   cd $iwd
}

opticks-cmake-modify-ex1(){
  local msg="=== $FUNCNAME : "
  local bdir=$(opticks-bdir)
  local bcache=$bdir/CMakeCache.txt
  [ ! -f "$bcache" ] && echo $msg requires a preexisting $bcache from prior opticks-cmake run && return 
  opticks-bcd
  g4-
  xercesc- 

  cmake \
       -DGeant4_DIR=$(g4-cmake-dir) \
       -DXERCESC_LIBRARY=$(xercesc-library) \
       -DXERCESC_INCLUDE_DIR=$(xercesc-include-dir) \
          . 
}

opticks-cmake-modify-ex2(){
  local msg="=== $FUNCNAME : "
  local bdir=$(opticks-bdir)
  local bcache=$bdir/CMakeCache.txt
  [ ! -f "$bcache" ] && echo $msg requires a preexisting $bcache from prior opticks-cmake run && return 
  opticks-bcd

  cmake \
       -DCOMPUTE_CAPABILITY=$(opticks-compute-capability) \
          . 
}

opticks-cmake-modify-ex3(){

  local msg="=== $FUNCNAME : "
  local bdir=$(opticks-bdir)
  local bcache=$bdir/CMakeCache.txt
  [ ! -f "$bcache" ] && echo $msg requires a preexisting $bcache from prior opticks-cmake run && return 
  opticks-bcd

  echo $msg opticks-cmakecache-vars BEFORE MODIFY 
  opticks-cmakecache-vars 

  cmake \
       -DOptiX_INSTALL_DIR=/Developer/OptiX_380 \
       -DCOMPUTE_CAPABILITY=30 \
          . 

  echo $msg opticks-cmakecache-vars AFTER MODIFY 
  opticks-cmakecache-vars 

}


opticks-configure()
{
   opticks-wipe

   case $(opticks-cmake-generator) in
       "Visual Studio 14 2015") opticks-configure-local-boost $* ;;
                             *) opticks-configure-system-boost $* ;;
   esac
}

opticks-configure-system-boost()
{
   opticks-cmake $* 
}

opticks-configure-local-boost()
{
    local msg="=== $FUNCNAME :"
    boost-

    local prefix=$(boost-prefix)
    [ ! -d "$prefix" ] && type $FUNCNAME && return  
    echo $msg prefix $prefix

    opticks-cmake \
              -DBOOST_ROOT=$prefix \
              -DBoost_USE_STATIC_LIBS=1 \
              -DBoost_USE_DEBUG_RUNTIME=0 \
              -DBoost_NO_SYSTEM_PATHS=1 \
              -DBoost_DEBUG=0 

    # vi $(cmake-find-package Boost)
}


#opticks-config-type(){ echo Debug ; }
opticks-config-type(){ echo RelWithDebInfo ; }

opticks--old(){     

   local msg="$FUNCNAME : "
   local iwd=$PWD

   local bdir=$1
   shift
   [ -z "$bdir" -o "$bdir" == "." ] && bdir=$(opticks-bdir) 
   [ ! -d "$bdir" ] && echo $msg bdir $bdir does not exist && return 

    
   cd $bdir

   cmake --build . --config $(opticks-config-type) --target ${1:-install}

   cd $iwd
}

opticks-t-old-approach(){  cat << EOA

   # this old approach was used before the move to 
   # treating the sub-projs as independant : so it 
   # worked from the top level bdir 

   local log=ctest.log
   #opticks-t-- $*

   date          | tee $log
   ctest $* 2>&1 | tee -a $log
   date          | tee -a $log

   cd $iwd
   echo $msg use -V to show output, ctest output written to $bdir/ctest.log


EOA
}


opticks-tl-old-approach()
{
   local arg=$1
   if [ "${arg:0:1}" == "/" -a -d "$arg" ]; then
       bdir=$arg
       shift
   else
       bdir=$(opticks-bdir) 
   fi
   ls -l $bdir/ctest.log
   cat $bdir/ctest.log
}




