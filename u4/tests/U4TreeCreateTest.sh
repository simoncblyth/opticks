#!/bin/bash -l 
usage(){ cat << EOU
U4TreeCreateTest.sh  : loads GDML, runs U4Tree::Create populating stree.h, saves to FOLD  
============================================================================================

::

    ~/o/u4/tests/U4TreeCreateTest.sh

Running this twice WITH_SND and NOT:WITH_SND enables 
comparison of snd.hh and sn.h based CSG node impls
as the CSG nodes are persisted into "stree/csg" WITH_SND 
and "stree/_csg" NOT:WITH_SND

NB when enabling/disabling SysRap WITH_SND need to rebuild sysrap+u4 
prior to re-running.  This is rather slow as changing flags forces
everything to be recompiled.  BUT: this is just transient whilst 
checking the newer more flexible sn.h impl

As soon are are satisfied that sn.h is behaving will no longer
need to switch back to WITH_SND. 

The test U4Polycone_test.sh avoids this slow rebuild by 
directly compiling the needed SysRap sources and not using 
the full SysRap library.  

EOU
}

SDIR=$(cd $(dirname $BASH_SOURCE) && pwd)

bin=U4TreeCreateTest 
defarg="info_run_ana"
arg=${1:-$defarg}

loglevels(){
   #export U4VolumeMaker=INFO
   #export U4Solid=INFO
   export DUMMY=INFO
}

loglevels


export U4TreeBorder__FLAGGED_ISOLID=HamamatsuR12860sMask_virtual0x61b0510
export U4Tree__IsFlaggedSolid_NAME=HamamatsuR12860sMask_virtual

source $HOME/.opticks/GEOM/GEOM.sh
gdmlpath=$HOME/.opticks/GEOM/$GEOM/origin.gdml

if [ ! -f "$gdmlpath" ]; then
   echo $BASH_SOURCE : ERROR GEOM $GEOM LACKS gdmlpath $gdmlpath 
   exit 1 
fi 
export ${GEOM}_GDMLPath=$gdmlpath

TMP=${TMP:-/tmp/$USER/opticks}
export FOLD=$TMP/$bin
script=$SDIR/$bin.py 

vars="BASH_SOURCE SDIR bin GEOM gdmlpath FOLD script"


if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%30s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 
fi 

if [ "${arg/build}" != "$arg" ]; then 
    echo $BASH_SOURCE : FATAL bin $bin IS BUILD BY STANDARD u4 OM : NOT THIS SCRIPT && exit 1
fi 

if [ "${arg/load}" != "$arg" ]; then 
    $bin load
    [ $? -ne 0 ] && echo $BASH_SOURCE load error && exit 2 
fi 

if [ "${arg/dbg}" != "$arg" ]; then 
    dbg__ $bin 
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 3
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 4
fi 

exit 0 


