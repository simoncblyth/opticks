#!/bin/bash 

usage(){ cat << EOU
CSGFoundryLoadTest.sh
========================



EOU

}


bin=CSGFoundryLoadTest
#source $OPTICKS_HOME/bin/GEOM_.sh 
source $HOME/.opticks/GEOM/GEOM.sh 

A_CFBaseFromGEOM=$HOME/.opticks/GEOM/$GEOM
B_CFBaseFromGEOM=/cvmfs/opticks.ihep.ac.cn/.opticks/GEOM/$GEOM

if [ -d "${A_CFBase}" ]; then
   export ${GEOM}_CFBaseFromGEOM=${A_CFBaseFromGEOM}
elif [ -d "${B_CFBase}" ]; then
   export ${GEOM}_CFBaseFromGEOM=${B_CFBaseFromGEOM}
fi 


export SSim__load_tree_load=1 

loglevels()
{
    export CSGFoundry=INFO
    export SSim=INFO
}
loglevels

defarg="run"
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


if [ "${arg/ana}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $(dirname $BASH_SOURCE)/$bin.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 3 
fi 

exit 0 


