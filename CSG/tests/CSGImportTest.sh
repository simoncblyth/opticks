#!/bin/bash -l 
usage(){ cat << EOU
CSGImportTest.sh : testing CSGFoundry::importTree 
===================================================

Workflow:

1. Create stree.h instance from loaded GDML and save it::

   cd ~/opticks/u4/tests  # u4t 
   ./U4TreeCreateTest.sh

2. Load stree.h instance and imports with CSGFoundry::importTree then saves CSGFoundry::

   cd ~/opticks/CSG/tests
   ./CSGImportTest.sh

See also::

   cd ~/opticks/sysrap/tests
   ./stree_load_test.sh run
   ./stree_load_test.sh ana


EOU
}

bin=CSGImportTest

export BASE=/tmp/$USER/opticks/U4TreeCreateTest
export FOLD=/tmp/$USER/opticks/$bin


check=$BASE/stree/nds.npy
if [ ! -f "$check" ]; then
   echo $BASH_SOURCE input stree does not exist at BASE $BASE check $check 
   exit 1 
fi 

mkdir -p $FOLD


loglevel(){
   export CSGImport=INFO
   export scsg_level=1

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

