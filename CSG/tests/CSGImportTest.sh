#!/bin/bash -l 
usage(){ cat << EOU
CSGImportTest.sh : testing CSGFoundry::importTree 
===================================================

Workflow:

1. Create stree.h instance from loaded GDML and save it::

   cd ~/opticks/u4/tests  # u4t 
   ./U4TreeCreateTest.sh

2. Load stree.h instance and imports with CSGFoundry::importTree then saves CSGFoundry::

   cd ~/opticks/CSG/tests     # ct 
   ./CSGImportTest.sh

* TODO: it would be more convenient to combine these two steps, ie go direct from gdml to imported stree CSGFoundry
* TODO: its a bit confusing that saving CSGFoundry saves stree also
   

3. The CSGFoundry persisted by CSGImportTest can be compated with others using python with::

   cd ~/opticks/CSG/tests     # ct 
   ./CSGFoundryAB.sh     
   

See also::

   cd ~/opticks/sysrap/tests  # st 
   ./stree_load_test.sh 
   ./stree_create_test.sh 

EOU
}

bin=CSGImportTest

source $HOME/.opticks/GEOM/GEOM.sh 
#export BASE=/tmp/$USER/opticks/U4TreeCreateTest
export BASE=/tmp/GEOM/$GEOM/CSGFoundry/SSim
export FOLD=/tmp/$USER/opticks/$bin


check=$BASE/stree/nds.npy
if [ ! -f "$check" ]; then
   echo $BASH_SOURCE input stree does not exist at BASE $BASE check $check 
   exit 1 
fi 

mkdir -p $FOLD


loglevel(){
   export CSGFoundry=INFO
   #export CSGImport=INFO
   #export scsg_level=1
   lvid=119
   ndid=0
   export LVID=${LVID:-$lvid}
   export NDID=${NDID:-$ndid}
}
loglevel

vars="BASH_SOURCE bin GEOM BASE FOLD check"

defarg=info_run
arg=${1:-$defarg}

if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 

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

