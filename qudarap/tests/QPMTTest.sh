#!/bin/bash -l 

usage(){ cat << EOU
QPMTTest
==========

NB : the test is not built by this script, use standard qudarap om to build it 


EOU
}

name=QPMTTest

defarg="run_ana"
arg=${1:-$defarg}

export QPMT=INFO
export FOLD=/tmp/$name

if [ "${arg/run}" != "$arg" ]; then  
    $name
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1
fi 

if [ "${arg/ana}" != "$arg" ]; then  
    ${IPYTHON:-ipython} --pdb -i $name.py 
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
