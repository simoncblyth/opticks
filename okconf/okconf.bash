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

okconf-source(){   echo $BASH_SOURCE ; }
okconf-vi(){       vi $(okconf-source) ; }
okconf-u(){  okconf-usage ; }
okconf-usage(){ cat << \EOU

OKConf : Opticks Configuration
=================================

Opticks is comprised of ~20 separate libraries which 
are configured via separate CMakeLists.txt. These 
libraries must be built in the correct order.

There is communication between the Opticks libs via
CMake generated configuration files.

OKConf is the very first library to be configured
and built, it looks for and determines versions 
and locations of:

1. CUDA
2. OptiX
3. Geant4

Generating a header 

    $(okconf-bdir)/inc/OKConf_Config.hh 

that gets installed into 

   $(opticks-prefix)/include/OKConf/OKConf_Config.hh

This header has defines for the CUDA, OptiX and Geant4
version integers allowing all Opticks code to know the 
versions of these externals at compile time.

FUNCTIONS
-----------

okconf---
    three dash form cleans first to force a full re-configuration

    Use this when switching between OptiX versions eg after changing 
    the override envvar:: 

        ## the location to look for OptiX defaults to $(opticks-prefix)/externals/OptiX
        ## to override that while testing another OptiX version set the below envvar 
        unset OPTICKS_OPTIX_PREFIX
        export OPTICKS_OPTIX_PREFIX=/usr/local/optix 


Related Notes
-----------------

* notes/issues/optix-version-switching.rst
* notes/issues/OpticksCUDAFlags.rst


EOU
}

okconf-env(){      olocal- ; opticks- ;  }


okconf-bdir(){ echo $(opticks-bdir)/okconf ; }
okconf-sdir(){ echo $(opticks-home)/okconf ; }
okconf-tdir(){ echo $(opticks-home)/okconf/tests ; }

okconf-bcd(){  cd $(okconf-bdir); }
okconf-scd(){  cd $(okconf-sdir); }
okconf-tcd(){  cd $(okconf-tdir); }

okconf-dir(){  echo $(okconf-sdir) ; }
okconf-cd(){   cd $(okconf-dir); }
okconf-c(){   cd $(okconf-dir); }


okconf-name(){ echo okconf ; }
okconf-tag(){  echo OKCONF ; }

okconf-apihh(){  echo $(okconf-sdir)/$(okconf-tag)_API_EXPORT.hh ; }
okconf---(){  
    local iwd=$PWD   

    okconf-cd 

    om-clean
    om-conf
    om-make    

    cd $iwd
}



okconf-wipe(){    local bdir=$(okconf-bdir) ; rm -rf $bdir ; } 

okconf--(){       opticks-- $(okconf-bdir) ; } 
okconf-t(){       opticks-t $(okconf-bdir) $* ; } 
okconf-gentest(){ okconf-tcd ; oks- ; oks-gentest ${1:-CExample} $(okconf-tag) ; } 
okconf-txt(){     vi $(okconf-sdir)/CMakeLists.txt $(okconf-tdir)/CMakeLists.txt ; } 



okconf-test-version-switching()
{
    local dir
    ls -1d /usr/local/OptiX_??? | while read dir ; do 
        unset OPTICKS_OPTIX_PREFIX
        export OPTICKS_OPTIX_PREFIX=$dir 
        okconf---
    done
    unset OPTICKS_OPTIX_PREFIX 
}



