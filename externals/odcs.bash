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

odcs-src(){      echo externals/odcs.bash ; }
odcs-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(odcs-src)} ; }
odcs-vi(){       vi $(odcs-source) ; }
odcs-env(){      olocal- ; opticks- ; }
odcs-usage(){ cat << EOU

DualContouringSample as Opticks External 
==========================================

See also env-;dcs-

NB uses same prefix as Opticks so that opticks/cmake/Modules/FindGLM.cmake succeeds


EOU
}

odcs-edit(){ vi $(opticks-home)/cmake/Modules/FindDualContouringSample.cmake ; }
odcs-url(){ echo https://github.com/simoncblyth/DualContouringSample ; }

odcs-dir(){  echo $(opticks-prefix)/externals/DualContouringSample/DualContouringSample ; }
odcs-bdir(){ echo $(opticks-prefix)/externals/DualContouringSample/DualContouringSample.build ; }

odcs-cd(){  cd $(odcs-dir); }
odcs-bcd(){ cd $(odcs-bdir) ; }

odcs-fullwipe()
{
    rm -rf $(opticks-prefix)/externals/DualContouringSample
    rm -f  $(opticks-prefix)/externals/lib/libDualContouringSample.dylib 
    rm -rf $(opticks-prefix)/externals/include/DualContouringSample
    ## test executables not removed
}

odcs-get(){
   local iwd=$PWD
   local dir=$(dirname $(odcs-dir)) &&  mkdir -p $dir && cd $dir
   [ ! -d DualContouringSample ] && git clone $(odcs-url)
   cd $iwd
}

odcs-cmake()
{
    local iwd=$PWD
    local bdir=$(odcs-bdir)

    mkdir -p $bdir
    #[ -f "$bdir/CMakeCache.txt" ] && echo $msg configured already && return  
    rm -f "$bdir/CMakeCache.txt"

    odcs-bcd   
    opticks-

    cmake \
       -DOPTICKS_PREFIX=$(opticks-prefix) \
       -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules \
       -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
       $* \
       $(odcs-dir)


    cd $iwd
}

odcs-make()
{
    local iwd=$PWD
    odcs-bcd
    cmake --build . --config Release --target ${1:-install}
    cd $iwd
}


odcs--()
{
   odcs-get
   odcs-cmake
   odcs-make install
}

odcs-t()
{
   odcs-make test
}

