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

tdefault-source(){   echo $(opticks-home)/tests/tdefault.bash ; }
tdefault-asource(){  echo $(opticks-home)/ana/tdefault.py ; }
tdefault-vi(){       vi $(tdefault-source) $(tdefault-asource) ; }
tdefault-usage(){ cat << \EOU

tdefault- 
==================================================


EOU
}
tdefault-env(){      olocal- ;  }
tdefault-dir(){ echo $(opticks-home)/tests ; }
tdefault-cd(){  cd $(tdefault-dir); }

join(){ local IFS="$1"; shift; echo "$*"; }

tdefault-tag(){ echo 1 ; }

tdefault--(){

    local msg="=== $FUNCNAME :"

    #local photons=1000000
    local photons=100000
    #local photons=20000
    #local photons=100

    local g4ppe=10000  # default 10k photons per g4 evt (subevt splitting for G4 memory reasons)
    case $photons in
       1|10|100|1000|10000) g4ppe=$photons ;;
     esac

    local tag=$(tdefault-tag)
    op.sh  \
            $* \
            --g4ppe $g4ppe \
            --animtimemax 15 \
            --timemax 15 \
            --eye 0,1,0 \
            --torch \
            --save --tag $tag --cat default \
            --rendermode +global,+axis,+in 



}
 
tdefault-args(){ echo --tag $(tdefault-tag) --det default --src torch ; } 
tdefault-i(){ ipython -i $(which tdefault.py) ; }
tdefault-distrib(){ ipython -i $(which tdefault_distrib.py) ; }

tdefault-t(){ tdefault-;tdefault-- --okg4 --compute $* ; } 
tdefault-d(){ tdefault-;tdefault-t --steppingdbg $* ; } 

tdefault-v(){   tdefault-;tdefault-- --okg4 --load $* ; }
tdefault-vg4(){ tdefault-;tdefault-- --okg4 --load --vizg4 $* ; }



