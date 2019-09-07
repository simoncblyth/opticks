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

tboxlaser-source(){   echo $(opticks-home)/tests/tboxlaser.bash ; }
tboxlaser-vi(){       vi $(tboxlaser-source) ; }
tboxlaser-usage(){ cat << \EOU

tboxlaser- : laser beam onto sample cube inside mineral oil box 
====================================================================









EOU
}
tboxlaser-env(){      olocal- ;  }
tboxlaser-dir(){ echo $(opticks-home)/tests ; }
tboxlaser-cd(){  cd $(tboxlaser-dir); }

join(){ local IFS="$1"; shift; echo "$*"; }

tboxlaser-tag(){ echo 1 ; }
tboxlaser-det(){ echo BoxInBox ; }

tboxlaser--(){
    local cmdline=$*
    local tag=$(tboxlaser-tag)
    local photons=500000

    #local nm=380
    local nm=480

    local m1=MineralOil
    #local m2=Pyrex
    local m2=GdDopedLS

    ## beam in -Z direction, fixpol +Y

    local torch_config=(
                 type=point
                 mode=fixpol
                 polarization=0,1,0
                 source=0,0,299
                 target=0,0,0
                 photons=$photons
                 material=$m1
                 wavelength=$nm
                 weight=1.0
                 time=0.1
                 zenithazimuth=0,1,0,1
                 radius=0
               )   
    local test_config=(
                 mode=BoxInBox
                 analytic=1

                 node=box
                 boundary=Rock//perfectAbsorbSurface/$m1
                 parameters=0,0,0,300

                 node=box
                 boundary=$m1///$m2
                 parameters=0,0,0,100
                   ) 

    op.sh \
       --test --testconfig "$(join _ ${test_config[@]})" \
       --torch --torchconfig "$(join _ ${torch_config[@]})" \
       --animtimemax 10 \
       --timemax 10 \
       --cat boxlaser --tag $tag --save  \
       --eye 0.5,0.5,0.0 \
       --rendermode +global,+axis \
       $* 
}
tboxlaser-args(){  echo  --tag $(tboxlaser-tag) --det boxlaser --src torch ; }
tboxlaser-a(){     tbox.py  $(tboxlaser-args) $* ; } 
tboxlaser-i(){     ipython -i $(which tbox.py) --  $(tboxlaser-args) $* ; } 
tboxlaser-t()
{
    tboxlaser-
    tboxlaser-- --okg4 --compute $*
}

tboxlaser-v()
{
    tboxlaser-- --okg4 --load $*
}

tboxlaser-vg4()
{
    tboxlaser-- --okg4 --vizg4 --load $*
}

tboxlaser-tfx()
{
    tboxlaser-t  --fxabconfig 10000 --fxab --fxscconfig 10000 --fxsc --fxreconfig 0.5 --fxre $*
}



