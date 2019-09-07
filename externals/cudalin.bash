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

cudalin-source(){   echo $BASH_SOURCE; }
cudalin-vi(){       vi $(cudalin-source) ; }
cudalin-env(){      olocal- ; }
cudalin-usage(){ cat << \EOU

CUDA on Linux : Version specific notes 
=========================================

See cuda- for more general info.

See Also
----------

cuda- 
   general CUDA notes
cudamac-
   version specifics on macOS


Overview on version dependencies
---------------------------------

Versions need to be carefully aligned, driven 
by OptiX, the release notes of which identify the
development CUDA version used and the minimum 
GPU driver version.  The CUDA version release notes
identify the minimum Linux kernel version.

* OptiX version 

  * GPU driver version (kernel extension, forcing stringent version alignment)

    * Linux kernel version 

  * CUDA version

    * CUDA driver version
    * Linux kernel version


nvidia display driver (aka GPU driver)
----------------------------------------

The GPU driver is normally provided by the vendor, 
BUT the CUDA driver requires a newer GPU driver than the old one
provided by the vendor/distribution.


Curious about how graphics drivers work
-----------------------------------------

* https://people.freedesktop.org/~marcheu/linuxgraphicsdrivers.pdf


GPU Drivers
-------------

Linux x64 (AMD64/EM64T) Display Driver : 418.43
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://www.nvidia.com/Download/driverResults.aspx/142958/en-us
 
Version:    418.43
Release Date:   2019.2.22
Operating System:   Linux 64-bit
Language:   English (US)
File Size:  101.71 MB
    

CUDA Environment
--------------------

::

    #export CUDA_VERSION=9.2
    export CUDA_VERSION=10.1
    export PATH=/usr/local/cuda-${CUDA_VERSION}/bin:${PATH}
    export LD_LIBRARY_PATH=/usr/local/cuda-${CUDA_VERSION}/lib64:${LD_LIBRARY_PATH}

EOU
}

cudalin-home(){  echo ${CUDALIN_HOME:-/usr/local/cuda} ; }
cudalin-name(){  echo $(basename $(readlink $(cudalin-home))) ; }
cudalin-vers(){  local name=$(cudalin-name) ; echo ${name/cuda-} ; }
cudalin-samples-dir(){ echo $LOCAL_BASE/NVIDIA_CUDA-$(cudalin-vers)_Samples ; }

cudalin-info(){ cat << EOI

    cudalin-home        : $(cudalin-home)
    cudalin-name        : $(cudalin-name)
    cudalin-vers        : $(cudalin-vers)
    cudalin-samples-dir : $(cudalin-samples-dir)

EOI
}

cudalin-samples-install()
{
    local dir=$(cudalin-samples-dir)
    local fold=$(dirname $dir)
    [ -d "$dir" ] && echo $FUNCNAME : The samples are already installed in $dir && return 
    [ -z "$fold" ] && echo $FUCNAME : ERROR no fold $fold && return

    cuda-install-samples-$(cudalin-vers).sh $dst
    ## hmm this is assuming the CUDA environment setup and the symbolic link match 
}

cudalin-scd(){ cd $(cudalin-samples-dir) ; }

cudalin-make(){
   cudalin-scd
   make -j$(nproc)
}

cudalin--()
{
    cudalin-samples-install
    cudalin-make
}

cudalin-sample(){ $(cudalin-samples-dir)/$* ; }
cudalin-deviceQuery(){ cudalin-sample 1_Utilities/deviceQuery/deviceQuery ; }




