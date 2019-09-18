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

trainbow-source(){ echo $BASH_SOURCE ; }
trainbow-dir(){    echo $(dirname $BASH_SOURCE) ; }
trainbow-vi(){     vi $(trainbow-source) ; }
trainbow-usage(){ cat << \EOU

trainbow- : Rainbow geometry with torch source
======================================================

`trainbow-pol s/p`
    Op and G4 simulations of rainbow geometry using S or P polarized torch source 
    followed by below -cf comparisons.

`trainbow-cf s/p`
    compare Opticks and Geant4 material/flag sequence histories and make 
    deviation angle plot.

    To see the plots use interactive ipython::

        In [1]: run trainbow.py --tag 6
        In [1]: run trainbow.py --tag 5


`trainbow-test`
    invokes `trainbow-pol` for both s and p  


ANALYSIS EXERCISE 
--------------------------

* start ipython from the `~/opticks/ana` directory::

       cd ~/opticks/ana
       ipython 

* run trainbow.py from within ipython, as described above,
  a plot should appear 

* modify the *trainbow.py* script to add extra plots

* interpret the meaning of the extra plots 

* if you have an full Opticks install, try modifying
  details of the simulation, run the simulation and
  check for changes in the plot




EOU
}

trainbow-env(){      olocal- ;  }
trainbow-cd(){  cd $(trainbow-dir); }

join(){ local IFS="$1"; shift; echo "$*"; }

trainbow-det(){  echo rainbow ; }
trainbow-src(){  echo torch ; }
trainbow-args(){ echo  --det $(trainbow-det) --src $(trainbow-src) ; }

trainbow-poltag()
{
   case $1 in 
     s) echo 5 ;;
     p) echo 6 ;;
   esac
}

trainbow--(){

    local msg="=== $FUNCNAME :"

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

    local tag=$(trainbow-poltag $pol)
    echo $msg pol $pol tag $tag 

    op.sh  \
            $cmdline \
            --animtimemax 10 \
            --timemax 10 \
            --geocenter \
            --eye 0,0,1 \
            --test --testconfig "$(trainbow-testconfig)" --dbganalytic \
            --torch --torchconfig "$(trainbow-torchconfig $pol)" \
            --tag $tag --cat $(trainbow-det) \
            --save \
            --rendermode +global,+axis

        #    --torchdbg \
}




trainbow-torchconfig()
{
    local pol=${1:-s}
    local wavelength=500
    local surfaceNormal=0,1,0
    #local azimuth=-0.25,0.25
    local azimuth=0,1
    local identity=1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000

    local photons=1000000
    #local photons=10000

    local torch_config=(
                 type=discIntersectSphere
                 photons=$photons
                 mode=${pol}pol
                 polarization=$surfaceNormal
                 frame=-1
                 transform=$identity
                 source=0,0,599
                 target=0,0,0
                 time=0.1
                 radius=100
                 distance=25
                 zenithazimuth=0,1,$azimuth
                 material=Vacuum
                 wavelength=$wavelength 
               )
    echo "$(join _ ${torch_config[@]})" 
}


trainbow-testconfig()
{
    #local material=GlassSchottF2
    local material=MainH2OHale

    local test_config=(
                 mode=BoxInBox
                 analytic=1

                 node=box      parameters=0,0,0,1200           boundary=Rock//perfectAbsorbSurface/Vacuum
                 node=sphere   parameters=0,0,0,100            boundary=Vacuum///$material
               )

     echo "$(join _ ${test_config[@]})" 
}


trainbow-cf() 
{       
   local pol=${1:-s}
   shift 
   local tag=$(trainbow-poltag ${pol})
   trainbow.py $(trainbow-args) --tag $tag $* 
} 

trainbow-icf(){   
   local pol=${1:-s}
   shift 
   local tag=$(trainbow-poltag ${pol}) 

   echo "enter: %debug " 
   ipython -i $(which trainbow.py) --  $(trainbow-args) --tag $tag $* 
}


trainbow-pol()
{
   local pol=${1:-s}
   shift 

   trainbow-- --${pol}pol --okg4 --compute
   trainbow-cf $pol
}


trainbow-t()
{
   trainbow-pol s
   trainbow-pol p
}

trainbow-vg4(){  trainbow-- $* --load --vizg4 ; } 
trainbow-v() {   trainbow-- $* --load ; } 




