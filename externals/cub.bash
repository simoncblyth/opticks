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

cub-source(){   echo ${BASH_SOURCE} ; }
cub-vi(){       vi $(cub-source) ; }
cub-usage(){ cat << EOU

CUB : CUDA Unbound
======================


* https://nvlabs.github.io/cub/






EOU
}


cub-edir(){ echo $(opticks-home)/numerics/cub ; }
cub-sdir(){ echo $(local-base)/env/numerics/cub ; }
cub-idir(){ echo $(cuda-idir)/cub ; }

cub-ecd(){  cd $(cub-edir) ; }
cub-scd(){  cd $(cub-sdir) ; }
cub-icd(){  cd $(cub-idir) ; }

cub-cd(){   cd $(cub-idir) ; }


cub-get()
{
    local dir=$(dirname $(cub-sdir)) &&  mkdir -p $dir && cd $dir
    [ ! -d cub ] && git clone git@github.com:simoncblyth/cub.git
}

cub-update()
{
    cub-scd
    git pull
}


cub-env(){      
   olocal- ; 
}

