#!/bin/bash -l 
usage(){ cat << EOU
amdahl.sh 
===========

::

    ~/o/ana/amdahl.sh ana
    ~/o/ana/amdahl.sh mpcap
    ~/o/ana/amdahl.sh mppub



EOU
}


cd $(dirname $(realpath $BASH_SOURCE))

defarg="info_ana"
arg=${1:-$defarg}
script=amdahl.py 

TMP=${TMP:-/tmp/$USER/opticks}
vars="BASH_SOURCE arg script TMP"

if [ "${arg/info}" != "$arg" ]; then
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi

if [ "${arg/ana}" != "$arg" ]; then
   ${IPYTHON:-ipython} --pdb -i $script 
   [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 1 
fi

if [ "$arg" == "mpcap" -o "$arg" == "mppub" ]; then
    export CAP_BASE=$TMP/figs
    export CAP_REL=amdahl 
    export CAP_STEM=amdahl
    case $arg in  
       mpcap) source mpcap.sh cap  ;;  
       mppub) source mpcap.sh env  ;;  
    esac
    if [ "$arg" == "mppub" ]; then 
        source epub.sh 
    fi  
fi 





exit 0 


