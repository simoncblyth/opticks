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

twhite-source(){ echo $BASH_SOURCE ; }
twhite-dir(){    echo $(dirname $BASH_SOURCE) ; }
twhite-vi(){     vi $(twhite-source) ; }
twhite-usage(){ cat << \EOU

twhite- : Pencil beam of white light incident on Glass prism
=================================================================

Pencil beam of white light incident on GlassSchottF2 prism.
This was based on tnewton- just changing to using a white light source.


`twhite-vi`
    edit the bash functions 

`twhite--`
    create Opticks geometry, simulates photons in interop mode, visualize, saves evt file 

`twhite-py s/p`
    plots simulated prism deviation angle against analytic expectation

`twhite-test`
    does both s and p simulation in compute mode and runs the py check 


EOU
}
twhite-env(){      olocal- ;  }
twhite-cd(){  cd $(twhite-dir); }

join(){ local IFS="$1"; shift; echo "$*"; }

twhite-det(){ echo white ; }
twhite-src(){ echo torch ; }
twhite-tag(){
    case ${1:-s} in  
        s) echo 1 ;;
        p) echo 2 ;;
    esac
}

twhite--(){

    local msg="=== $FUNCNAME :"
    local cmdline=$*
    local det=$(twhite-det)

    local pol
    if [ "${cmdline/--spol}" != "${cmdline}" ]; then
         pol=s
         cmdline=${cmdline/--spol}
    elif [ "${cmdline/--ppol}" != "${cmdline}" ]; then
         pol=p
         cmdline=${cmdline/--ppol}
    else
         pol=s
    fi  

    local tag=$(twhite-tag $pol)
    echo  $msg pol $pol tag $tag

    local material=GlassSchottF2
    local surfaceNormal=0,1,0

    local torch_config=(
                 type=point
                 photons=500000
                 mode=${pol}pol,wavelengthSource
                 polarization=$surfaceNormal
                 frame=-1
                 transform=0.500,0.866,0.000,0.000,-0.866,0.500,0.000,0.000,0.000,0.000,1.000,0.000,-86.603,0.000,0.000,1.000
                 source=-200,200,0
                 target=0,0,0
                 radius=50
                 distance=25
                 zenithazimuth=0,1,0,1
                 material=Vacuum
                 wavelength=0 
               )
 
    local test_config=(
                 mode=BoxInBox
                 analytic=1
                 node=box   parameters=-1,1,0,700       boundary=Rock//perfectAbsorbSurface/Vacuum
                 node=prism parameters=60,300,300,200   boundary=Vacuum///$material
               )

    op.sh  \
            $* \
            --animtimemax 7 \
            --timemax 7 \
            --geocenter \
            --eye 0,0,1 \
            --test --testconfig "$(join _ ${test_config[@]})" \
            --torch --torchconfig "$(join _ ${torch_config[@]})" \
            --torchdbg \
            --save --tag $tag --cat $det \
            --rendermode +global,+axis

}

twhite-args(){ echo --tag $(twhite-tag ${1:-s}) --det $(twhite-det) --src $(twhite-src) ;  }
twhite-ipy(){  ipython -i $(which twhite.py) -- $(twhite-args $1) ; }
twhite-py(){   twhite.py $(twhite-args $1) ; }

twhite-t()
{
    twhite-- --spol --compute
    twhite-- --ppol --compute
}

