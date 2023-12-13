#!/bin/bash -l 
usage(){ cat << EOU
CSGTargetTest.sh
==================

~/o/CSG/tests/CSGTargetTest.sh 


EOU
}

SDIR=$(dirname $(realpath $BASH_SOURCE))
source $HOME/.opticks/GEOM/GEOM.sh 

vars="BASH_SOURCE name script LOG arg SDIR METHOD MOI GEOM" 

#moi=Hama:0:1000
moi=NNVT:0:1000
export MOI=${MOI:-$moi}
export METHOD=getFrame

logging()
{
    export CSGTarget=INFO
}
[ -n "$LOG" ] && logging


name=CSGTargetTest
script=$SDIR/$name.py 

defarg=info_run
arg=${1:-$defarg}


if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%25s : %s \n" "$var" "${!var}" ; done 
fi

if [ "${arg/run}" != "$arg" ]; then 
    $name
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 
fi 

if [ "${arg/dbg}" != "$arg" ]; then 
    dbg__ $name
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 2 
fi 

if [ "${arg/ana}" != "ana" ]; then 
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 3 
fi 

exit 0 

