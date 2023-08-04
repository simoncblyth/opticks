#!/bin/bash -l 
usage(){ cat << EOU
G4CXAppTest.sh 
================



EOU
}

SDIR=$(cd $(dirname $BASH_SOURCE) && pwd)
U4TDIR=$(cd $SDIR/../../u4/tests && pwd)

bin=G4CXAppTest

defarg="info_run_ana"
#defarg="info_dbg_ana"
arg=${1:-$defarg}

source $HOME/.opticks/GEOM/GEOM.sh 

geomscript=$U4TDIR/$GEOM.sh
if [ -f "$geomscript" ]; then  
    source $geomscript
else
    echo $BASH_SOURCE : no geomscript $geomscript
fi 
ana=$SDIR/$bin.py 


export BASE=/tmp/$USER/opticks/GEOM/$GEOM/$bin
export FOLD=$BASE/ALLVERSION/p001



#num_photons=1
#num_photons=10
#num_photons=100
#num_photons=1000      # 1k
num_photons=10000    # 10k
#num_photons=50000    # 50k 
#num_photons=100000   # 100k
#num_photons=1000000  # 1M

NUM_PHOTONS=${NUM_PHOTONS:-$num_photons}


export G4CXApp__PRIMARY_MODE=torch
export OPTICKS_MAX_BOUNCE=31  
export OPTICKS_EVENT_MODE=StandardFullDebug

export SEvent_MakeGensteps_num_ph=${NUM_PHOTONS}
source $U4TDIR/storch_FillGenstep.sh
env | grep storch


vars="BASH_SOURCE SDIR U4TDIR GEOM bin geomscript BASE FOLD ana" 

if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 1 
fi 

if [ "${arg/dbg}" != "$arg" ]; then
    case $(uname) in 
       Darwin) lldb__ $bin ;;
       Linux)  gdb__  $bin ;;
    esac
    [ $? -ne 0 ] && echo $BASH_SOURCE : dbg error && exit 2 
fi 

if [ "${arg/ana}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $ana
    [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 3
fi 

exit 0 


