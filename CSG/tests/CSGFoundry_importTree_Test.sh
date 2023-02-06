#!/bin/bash -l 
usage(){ cat << EOU
CSGFoundry_importTree_Test.sh : testing CSGFoundry::importTree CSGMaker::importTree
=====================================================================================

Workflow:

1. Create stree.h instance from loaded GDML and save it::

   cd ~/opticks/u4/tests  # u4t 
   ./U4TreeCreateTest.sh

2. Load the stree.h instance and test CSGFoundry::importTree CSGMaker::importTree::

   cd ~/opticks/CSG/tests
   ./CSGFoundry_importTree_Test.sh

See also::

   cd ~/opticks/sysrap/tests
   ./stree_load_test.sh run
   ./stree_load_test.sh ana


EOU
}

bin=CSGFoundry_importTree_Test

export BASE=/tmp/$USER/opticks/U4TreeCreateTest

loglevel(){
   export CSGImport=INFO
}
loglevel


defarg=run
arg=${1:-$defarg}

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1
fi 

if [ "${arg/dbg}" != "$arg" ]; then 
    case $(uname) in
       Darwin) lldb__ $bin ;;
       Linux)  gdb__  $bin ;; 
    esac
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 2
fi 

exit 0

