#!/bin/bash -l 
usage(){ cat << EOU
QPropTest.sh
=============



EOU
}

source $HOME/.opticks/GEOM/GEOM.sh 
SDIR=$(cd $(dirname $BASH_SOURCE) && pwd )

name=QPropTest 

export FOLD=${TMP:-/tmp/$USER/opticks}/$name
mkdir -p $FOLD/float
mkdir -p $FOLD/double

defarg="info_run_ana"
arg=${1:-$defarg}

vars="FOLD GEOM name" 


loglevels(){ 
   export SProp=INFO
}
loglevels


if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}"  ; done 
fi

if [ "${arg/run}" != "$arg" ]; then 
    $name
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 1  
fi 

if [ "${arg/dbg}" != "$arg" ]; then 
    case $(uname) in
       Darwin) lldb__ $name ;;
       Linux)  gdb__ $name ;;
    esac
    [ $? -ne 0 ] && echo $BASH_SOURCE : dbg error && exit 2  
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $SDIR/$name.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 2
fi 



