#!/bin/bash -l 
usage(){ cat << EOU
QPMTTest
==========

NB : standard qudarap/om builds the QPMTTest binary, not this script

EOU
}

REALDIR=$(cd $(dirname $BASH_SOURCE) && pwd)
name=QPMTTest

defarg="run_ana"
arg=${1:-$defarg}

logging(){
   export QPMT=INFO
}
logging

export FOLD=/tmp/$name
vars="REALDIR FOLD name"

if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%30s : %s \n" "$var" "${!var}" ; done 
fi
 
if [ "${arg/run}" != "$arg" ]; then  
    $name
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1
fi 

if [ "${arg/dbg}" != "$arg" ]; then  
    case $(uname) in
       Darwin) lldb__ $name ;;
       Linux) gdb__ $name ;;
    esac
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1
fi 

if [ "${arg/ana}" != "$arg" ]; then  
    ${IPYTHON:-ipython} --pdb -i $REALDIR/$name.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 2 
fi 

if [ "$arg" == "mpcap" -o "$arg" == "mppub" ]; then
    export CAP_BASE=$FOLD/figs
    export CAP_REL=QPMTTest
    export CAP_STEM=QPMTTest
    case $arg in  
       mpcap) source mpcap.sh cap  ;;  
       mppub) source mpcap.sh env  ;;  
    esac

    if [ "$arg" == "mppub" ]; then 
        source epub.sh 
    fi  
fi 

exit 0 
