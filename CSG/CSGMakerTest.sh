#!/bin/bash -l 

usage(){ cat << EOU
CSGMakerTest.sh : Creates CSGFoundry directories of CSGSolid/CSGPrim/CSGNode using CSG/tests/CSGMakerTest.cc
===============================================================================================================

Used to create small test geometries, often with single solids.::

   cd ~/opticks/CSG         ## OR "c" shortcut function

   vi ~/.opticks/GEOM.txt   ## OR "geom" shortcut function 
                            ## uncomment or add GEOM name with projection suffix _XY etc..  

   ./CSGMakerTest.sh        ## sources bin/GEOM.sh to set GEOM envvar and runs CSGMakerTest to create the CSGFoundry  

Subsequenly visualize the geometry with::

    cd ~/opticks/CSGOptiX   ## OR "cx" shortcut 
    EYE=-1,-1,-1 ./cxr_geochain.sh       ##  reads the GEOM.txt file to pick the CSGFoundry to load

EOU
}

source $(dirname $BASH_SOURCE)/../bin/GEOM.sh trim 

export CSGMaker_makeBoxedSphere_FACTOR=10

bin=CSGMakerTest 
echo === $BASH_SOURCE :  GEOM $GEOM bin $bin which $(which $bin)
defarg="run"
arg=${1:-$defarg}

if [ "$arg" == "run" ]; then 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 
fi 

if [ "$arg" == "dbg" ]; then 
    case $(uname) in 
      Darwin) lldb__ $bin ;;
      Linux)  gdb__ $bin ;; 
    esac
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 2 
fi 

exit 0
