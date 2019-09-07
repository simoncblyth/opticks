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

ttemplate-source(){   echo $(opticks-home)/tests/ttemplate.bash ; }
ttemplate-vi(){       vi $(ttemplate-source) ; }
ttemplate-usage(){ cat << \EOU

ttemplate- : Integration Tests Template
================================================


`ttemplate-vi`
    edit the bash functions 

`ttemplate--`
    create Opticks geometry, simulates photons in interop mode, visualize, saves evt file 

`ttemplate-- --compute`
    create Opticks geometry with OKTest, simulates, saves

`ttemplate-- --okg4` 
    create Opticks and Geant4 geometry with OKG4Test executable, simulates, saves 

`ttemplate-- --vizg4 --load`
    visualize the Geant4 simulated photon propagation 

`ttemplate-cf`
    compare Opticks and Geant4 material/flag sequence histories


`ttemplate-t`
    simulates with Opticks and Geant4 and compares the results 



EOU
}
ttemplate-env(){      olocal- ;  }
ttemplate-dir(){ echo $(opticks-home)/tests ; }
ttemplate-cd(){  cd $(ttemplate-dir); }

join(){ local IFS="$1"; shift; echo "$*"; }

ttemplate-tag(){ echo 10 ; }
ttemplate-det(){ echo PmtInBox ; }
ttemplate-src(){ echo torch ; }

ttemplate--(){
   type $FUNCNAME

    local msg="=== $FUNCNAME :"

    local cmdline=$*
    local tag=$(ttemplate-tag)
    local det=$(ttemplate-det)

    #local photons=500000
    local photons=100000

    local zenith=0,1
    #local typ=disclin
    local typ=disc
    local src=0,0,300
    local tgt=0,0,0
    local radius=100

    local mode=""
    local polarization=""

    local torch_config=(
                 type=$typ
                 photons=$photons
                 wavelength=380 
                 frame=1
                 source=$src
                 target=$tgt
                 radius=$radius
                 zenithazimuth=$zenith,0,1
                 material=Vacuum

                 mode=$mode
                 polarization=$polarization
               )


    local groupvelkludge=0
    local testverbosity=1
    local test_config=(
                 mode=PmtInBox
                 pmtpath=$OPTICKS_INSTALL_PREFIX/opticksdata/export/dpib/GMergedMesh/0
                 control=$testverbosity,0,0,0
                 analytic=1
                 groupvel=$groupvelkludge
                 shape=box
                 boundary=Rock/NONE/perfectAbsorbSurface/MineralOil
                 parameters=0,0,0,300
                   ) 


   op.sh \
       --test --testconfig "$(join _ ${test_config[@]})" \
       --torch --torchconfig "$(join _ ${torch_config[@]})" \
       --timemax 10 \
       --animtimemax 10 \
       --cat $det --tag $tag --save \
       --eye 0.0,-0.5,0.0 \
       --geocenter \
       $* 

}


ttemplate-args() {        echo  --tag $(ttemplate-tag) --det $(ttemplate-det) --src $(ttemplate-src) ; }
ttemplate-cf() {          ttemplate.py          $(ttemplate-args) ; } 
ttemplate-cf-distrib() {  ttemplate_distrib.py  $(ttemplate-args) ; } 

ttemplate-ana()
{
    ttemplate-cf
    ttemplate-cf-distrib
}

ttemplate-test()
{
    ttemplate--  --okg4 --compute 
    ttemplate-cf
    ttemplate-cf-distrib
}

ttemplate-vg4() {  ttemplate-- --load --vizg4 ; } 
ttemplate-v() {    ttemplate-- --load ; } 


