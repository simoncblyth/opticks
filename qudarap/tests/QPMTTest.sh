#!/bin/bash
usage(){ cat << EOU
QPMTTest
==========

NB : standard qudarap/om builds the QPMTTest binary, not this script

::

   PLOT=qeshape ~/o/qudarap/tests/QPMTTest.sh pdb
   PLOT=cetheta ~/o/qudarap/tests/QPMTTest.sh pdb

   OPT=A_,Aa,As,Ap ./QPMTTest.sh ana
   OPT=R_,Ra,Rs,Rp ./QPMTTest.sh ana

   OPT=R_,Ra,Rs,Rp,A_,Aa,As,Ap,T_,Ta,Ts,Tp ./QPMTTest.sh ana


EOU
}

cd $(dirname $(realpath $BASH_SOURCE))
SCRIPT=$(basename $BASH_SOURCE)
export SCRIPT

name=QPMTTest
script=$name.py

source $HOME/.opticks/GEOM/GEOM.sh # define GEOM envvar

defarg="info_run_ana"
arg=${1:-$defarg}

logging(){
   type $FUNCNAME
   export QPMT=INFO
   #export QU=INFO
}
[ -n "$LOG" ] && logging

export FOLD=${TMP:-/tmp/$USER/opticks}/$name
vars="PWD FOLD GEOM ${GEOM}_CFBaseFromGEOM name"

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%30s : %s \n" "$var" "${!var}" ; done
fi

if [ "${arg/build}" != "$arg" ]; then
   echo $BASH_SOURCE : ERROR : QPMTTest IS BUILT BY STANDARD OM : NOT THIS SCRIPT && exit 1
fi

if [ "${arg/run}" != "$arg" ]; then
    $name
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1
fi

if [ "${arg/ls}" != "$arg" ]; then
    echo ls -alst $FOLD
    ls -alst $FOLD
fi


if [ "${arg/dbg}" != "$arg" ]; then
    source dbg__.sh
    dbg__ $name
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1
fi

if [ "${arg/pdb}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $BASH_SOURCE pdb error && exit 2
fi

if [ "${arg/ana}" != "$arg" ]; then
    ${PYTHON:-python} $script
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
