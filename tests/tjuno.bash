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

tjuno-source(){   echo $(opticks-home)/tests/tjuno.bash ; }
tjuno-vi(){       vi $(tjuno-source) ; }
tjuno-usage(){ cat << \EOU
tjuno- 
======================================================
 
FUNCTIONS
----------
EOU
}

tjuno-env(){      olocal- ;  }
tjuno-dir(){ echo $(opticks-home)/tests ; }
tjuno-cd(){  cd $(tjuno-dir); }

join(){ local IFS="$1"; shift; echo "$*"; }

tjuno-tag(){  echo 1 ; }
tjuno-det(){  echo j1707 ; }
tjuno-src(){  echo torch ; }
tjuno-args(){ echo  --det $(tjuno-det) --src $(tjuno-src) ; }

tjuno--(){

    tjuno-

    local msg="=== $FUNCNAME :"
    local cmdline=$*

    #local stack=4096
    local stack=2180  # default

    op.sh  \
            $cmdline \
            --j1707 --gltf 3 \
            --animtimemax 200 \
            --timemax 200 \
            --geocenter \
            --stack $stack \
            --eye 1,0,0 \
            --dbganalytic \
            --torch --torchconfig "$(tjuno-torchconfig)" \
            --torchdbg \
            --tag $(tjuno-tag) --cat $(tjuno-det) \
            --rendermode +global,+axis,+in \
            --save 
}

tjuno-tracetest()
{
    tjuno-- --tracetest $*
}

tjuno-torchconfig()
{

    local pol=${1:-s}
    local wavelength=500
    local identity=1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000

    #local photons=1000000
    local photons=100000
    #local photons=1

    local torch_config_disc=(
                 type=disc
                 photons=$photons
                 mode=fixpol
                 polarization=1,1,0
                 frame=-1
                 transform=$identity
                 source=0,0,599
                 target=0,0,0
                 time=0.1
                 radius=300
                 distance=200
                 zenithazimuth=0,1,0,1
                 material=Vacuum
                 wavelength=$wavelength 
               )


    local discaxial_target=0,0,0
    local torch_config_discaxial=(
                 type=discaxial
                 photons=$photons
                 frame=12
                 transform=$identity
                 source=$discaxial_target
                 target=0,0,0
                 time=0.1
                 radius=100
                 distance=400
                 zenithazimuth=0,1,0,1
                 material=Vacuum
                 wavelength=$wavelength 
               )



    local torch_config_sphere=(
                 type=sphere
                 photons=10000
                 frame=12
                 transform=1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,1000.000,1.000
                 source=0,0,0
                 target=0,0,1
                 time=0.1
                 radius=100
                 distance=400
                 zenithazimuth=0,1,0,1
                 material=LS
                 wavelength=$wavelength 
               )



    #echo "$(join _ ${torch_config_discaxial[@]})" 
    #echo "$(join _ ${torch_config_disc[@]})" 
    echo "$(join _ ${torch_config_sphere[@]})" 
}


tjuno-validate() {
    SRC=${1:-cerenkov}; shift
    TAG=${1:-1}; shift

    op.sh --j1707 --gltf 3 --${SRC} --tag ${TAG} --compute --save
    tevt.py --det juno1707 --src ${SRC} --tag ${TAG}

}
