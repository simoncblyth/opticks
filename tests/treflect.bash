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

treflect-source(){   echo $(opticks-home)/tests/treflect.bash ; }
treflect-vi(){       vi $(treflect-source) ; }
treflect-usage(){ cat << \EOU

treflect- : Fresnel reflection vs incident angle check  
==========================================================

A hemi-spherical S/P polarized light source focussed on cube face
is used to check the amount of reflection as a function of incident
angle matches expectation of the Fresnel formula. 


`treflect--`
    create Opticks geometry, simulates photons in interop mode, saves evt file 


EXERCISE
-----------

* run python analysis :doc:`../ana/treflect` using ipython,
  interpret the plot obtained, adjust parameters of the 
  simulation and repeat

* calculate a chi2 comparing analytic Fresnel formula expectations 
  with simulation results for S and P polarization, 
  use this to implement a python test that fails when they disagree
 
For background on analysis tools see :doc:`../ana/tools` 


EOU
}
treflect-env(){      olocal- ;  }
treflect-dir(){ echo $(opticks-home)/tests ; }
treflect-cd(){  cd $(treflect-dir); }

join(){ local IFS="$1"; shift; echo "$*"; }

treflect-det(){ echo reflect ; }
treflect-src(){ echo torch ; }


treflect-medium(){ echo Vacuum ; }

#treflect-material(){ echo GlassSchottF2 ; }
treflect-material(){ echo MainH2OHale ; }

trelect-container(){ echo Rock//perfectAbsorbSurface/$(treflect-medium) ; }
treflect-object(){ echo $(treflect-medium)///$(treflect-material) ; }

treflect-testconfig()
{
    local test_config=(
                 name=$FUNCNAME
                 mode=BoxInBox
                 analytic=1
                 node=box   parameters=0,0,0,1000       boundary=$(trelect-container)
                 node=box   parameters=0,0,0,200        boundary=$(treflect-object)
               )
     echo "$(join _ ${test_config[@]})" 
}

treflect-torchconfig()
{
    local pol=$1
    local photons=1000000
    # target is ignored for refltest, source is the focus point 

    local torch_config=(
                 type=refltest
                 photons=$photons
                 mode=${pol}pol,flatTheta
                 polarization=0,0,-1
                 frame=-1
                 transform=1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000
                 source=0,0,-200
                 radius=100
                 distance=25
                 zenithazimuth=0.5,1,0,1
                 material=$(treflect-medium)
                 wavelength=550
               )

     echo "$(join _ ${torch_config[@]})" 
}



treflect--()
{
    type $FUNCNAME

    local cmdline=$*
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

    case $pol in  
        s) tag=1 ;;
        p) tag=2 ;;
    esac

    echo  pol $pol tag $tag


    local testconfig
    if [ -n "$TESTCONFIG" ]; then
        testconfig=${TESTCONFIG}
    else
        testconfig=$(treflect-testconfig)
    fi

    local torchconfig
    if [ -n "$TORCHCONFIG" ]; then
        torchconfig=${TORCHCONFIG}
    else
        torchconfig=$(treflect-torchconfig $pol)
    fi

    op.sh  \
            $* \
            --animtimemax 7 \
            --timemax 7 \
            --geocenter \
            --eye 0,0,1 \
            --test --testconfig "$testconfig" \
            --torch --torchconfig "$torchconfig" \
            --torchdbg \
            --save --tag $tag --cat $(treflect-det) \
            --rendermode +global
  

}


treflect-args(){ echo --stag 1 --ptag 2 --det $(treflect-det) --src $(treflect-src) ; }
treflect-py(){   treflect.py $(treflect-args) $* ; } 
treflect-ipy(){  ipython -i $(which treflect.py) -- $(treflect-args) $* ; }

treflect-t()
{
    treflect--  --spol --compute
    treflect--  --ppol --compute
    treflect-py 
}

treflect-v()
{
    treflect-- --load $* --fullscreen
}



