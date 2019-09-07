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

tboolean-bib-source(){   echo $(opticks-home)/tests/tboolean-bib.bash ; }
tboolean-bib-vi(){       vi $(tboolean-bib-source) ; }
tboolean-bib-usage(){ cat << \EOU

tboolean-bib
================

NB : these funcs are parasitic wrt tboolean--



*tboolean-bib* uses the old BoxInBox or PmtInBox 
approach to specifying test geometry. 

This approach it now seldom used : so highly likely to be broken.

However this is still retained as it is expected that the 
old partition-at-intersection approach to geometry is actually 
faster than the new fully general CSG approach. 



PmtInBox

     * see tpmt- for this one

BoxInBox

     * CSG combinations not supported, union/intersection/difference nodes
       appear as placeholder boxes

     * raytrace superficially looks like a union, but on navigating inside 
       its apparent that its just overlapped individual primitives




Historical Note
------------------

CsgInBox

     * DECLARED DEAD, USE PyCsgInBox
     * requires "offsets" identifying node splits into primitives eg offsets=0,1 
     * nodes are specified in tree levelorder, trees must be perfect 
       with 1,3,7 or 15 nodes corresponding to trees of height 0,1,2,3




EOU
}


tboolean-bib-env(){      olocal- ; tboolean- ;   }
tboolean-bib-dir(){ echo $(opticks-home)/tests ; }
tboolean-bib-cd(){  cd $(tboolean-bib-dir); }

join(){ local IFS="$1"; shift; echo "$*"; }




tboolean-bib-box-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-bib-box(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null) tboolean-- $* ; }
tboolean-bib-box-()
{
    local test_config=(
                 mode=BoxInBox
                 name=$FUNCNAME
                 analytic=1

                 node=box      parameters=0,0,0,1000               boundary=$(tboolean-container)
                 node=box      parameters=0,0,0,100                boundary=$(tboolean-testobject)

                    )
     echo "$(join _ ${test_config[@]})" 
}
tboolean-bib-box--(){ echo -n ; }


tboolean-bib-box-small-offset-sphere-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-bib-box-small-offset-sphere(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null) tboolean-- $* ; }
tboolean-bib-box-small-offset-sphere-()
{
    local test_config=(
                 mode=BoxInBox
                 name=$FUNCNAME
                 analytic=1

                 node=sphere           parameters=0,0,0,1000          boundary=$(tboolean-container)
 
                 node=${1:-difference} parameters=0,0,0,300           boundary=$(tboolean-testobject)
                 node=box              parameters=0,0,0,200           boundary=$(tboolean-testobject)
                 node=sphere           parameters=0,0,200,100         boundary=$(tboolean-testobject)
               )
     echo "$(join _ ${test_config[@]})" 
}
tboolean-bib-box-small-offset-sphere--(){ echo -n ; }


tboolean-bib-box-sphere-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-bib-box-sphere(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null) tboolean-- $* ; }
tboolean-bib-box-sphere-()
{
    local operation=${1:-difference}
    local inscribe=$(python -c "import math ; print 1.3*200/math.sqrt(3)")
    local test_config=(
                 mode=BoxInBox
                 name=$FUNCNAME
                 analytic=1

                 node=box          parameters=0,0,0,1000          boundary=$(tboolean-container)
 
                 node=$operation   parameters=0,0,0,300           boundary=$(tboolean-testobject)
                 node=box          parameters=0,0,0,$inscribe     boundary=$(tboolean-testobject)
                 node=sphere       parameters=0,0,0,200           boundary=$(tboolean-testobject)
               )

     echo "$(join _ ${test_config[@]})" 
}
tboolean-bib-box-sphere--(){ echo -n ; }


