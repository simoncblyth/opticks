#!/bin/bash
usage(){ cat << EOU
U4TreeCreateTest.sh  : loads geometry, runs U4Tree::Create populating stree.h, saves to FOLD
==============================================================================================

The geometry may be loaded by various means including from GDML or j/PMTSim via U4VolumeMaker::

    ~/o/u4/tests/U4TreeCreateTest.sh

To update the stree impl used by U4TreeCreateTest it is necessary to
build and install both sysrap and u4::

     sy;om;u4;om

See also U4Polycone_test.sh which avoids slow rebuild after changing WITH macros
by directly compiling the needed SysRap sources and not using the full SysRap library.

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))
SDIR=$PWD

bin=U4TreeCreateTest
script=$bin.py

export TMP=${TMP:-/tmp/$USER/opticks}
export FOLD=$TMP/$bin



defarg="info_run_ana"
arg=${1:-$defarg}

logging()
{
   type $FUNCNAME
   export U4VolumeMaker=INFO
   export U4Solid=INFO
   #export stree__level=1
   export DUMMY=INFO
}
[ -n "$LOG" ] && logging


# extra logging for some bits of geometry ?
#export U4TreeBorder__FLAGGED_ISOLID=HamamatsuR12860sMask_virtual0x61b0510
export U4Solid__IsFlaggedLVID=0
#export U4Solid__IsFlaggedName=sDeadWater


source $HOME/.opticks/GEOM/GEOM.sh

_CFB=${GEOM}_CFBaseFromGEOM
CFB=${!_CFB}
xgdmlpath=$CFB/origin.gdml

if [ -n "$CFB" -a -d "$CFB" -a -f "$CFB/origin.gdml" ]; then
    note="binary $bin is expected to load $xgdmlpath"
else
    note="NOT using CFBaseFromGEOM resolution"
fi


vars="BASH_SOURCE SDIR defarg arg bin GEOM _CFB CFB xgdmlpath note TMP FOLD script"


if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%30s : %s \n" "$var" "${!var}" ; done
fi

if [ "${arg/clean}" != "$arg" ]; then
    cd $TMP && rm -rf U4TreeCreateTest  # hardcode directory name for safety
    [ $? -ne 0 ] && echo $BASH_SOURCE clean error && exit 1
    cd $SDIR
fi

if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1
fi

if [ "${arg/load}" != "$arg" ]; then
    $bin load
    [ $? -ne 0 ] && echo $BASH_SOURCE load error && exit 2
fi

if [ "${arg/dbg}" != "$arg" ]; then
    source dbg__.sh
    dbg__ $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 3
fi

if [ "${arg/ana}" != "$arg" ]; then
    ${PYTHON:-python} $script
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 4
fi

if [ "${arg/pdb}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $BASH_SOURCE pdb error && exit 5
fi

exit 0
