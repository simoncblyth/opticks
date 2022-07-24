#!/bin/bash -l 
usage(){ cat << EOU
gxt.sh : G4CXSimtraceTest 
=============================================================================================================

::

    cd ~/opticks/g4cx   # gx
    ./gxt.sh 
    ./gxt.sh info
    ./gxt.sh fold
    ./gxt.sh run
    ./gxt.sh dbg
    ./gxt.sh grab
    ./gxt.sh ana

To capture pyvista or matplotlib screens:: 

    ./gxt.sh pvcap
    ./gxt.sh pvpub           # check the paths used by pvpub
    PUB=chk ./gxt.sh pvpub   # copy into /env/presentation 

    ./gxt.sh mpcap
    ./gxt.sh mppub           # check the paths used by mppub
    PUB=chk ./gxt.sh mppub   # copy into /env/presentation 


As B uses A and T uses A+B the running order is:

A. gx ; ./gxs.sh 
B. u4 ; ./u4s.sh 
T. gx ; ./gxt.sh 

EOU
}

msg="=== $BASH_SOURCE :"

case $(uname) in 
  Linux)  defarg="run"  ;;
  Darwin) defarg="ana"  ;;
esac

arg=${1:-$defarg}

case $arg in
  fold) QUIET=1 ;;
esac

bin=G4CXSimtraceTest
gxtdir=$(dirname $BASH_SOURCE)
source $gxtdir/../bin/COMMON.sh 

BASE=$GEOMDIR/$bin
UBASE=${BASE//$HOME\/}    # UBASE relative to HOME to handle rsync between different HOME
FOLD=$BASE/ALL            # corresponds SEvt::save() with SEvt::SetReldir("ALL")


upfind_cfbase(){
    : traverse directory tree upwards searching CFBase geometry dir identified by existance of CSGFoundry/solid.npy  
    local dir=$1
    while [ ${#dir} -gt 1 -a ! -f "$dir/CSGFoundry/solid.npy" ] ; do dir=$(dirname $dir) ; done 
    echo $dir
}


# analysis/plotting uses A_FOLD B_FOLD for comparison together with the simtrace 

T_FOLD=$FOLD
A_FOLD=$($OPTICKS_HOME/g4cx/gxs.sh fold)
B_FOLD=$($OPTICKS_HOME/u4/u4s.sh fold)

T_CFBASE=$(upfind_cfbase $T_FOLD)
A_CFBASE=$(upfind_cfbase $A_FOLD)  
B_CFBASE=$(upfind_cfbase $B_FOLD)  


export A_FOLD
export A_CFBASE
export B_FOLD

if [ "${arg/info}" != "$arg" ]; then 
    vars="BASH_SOURCE arg bin defarg gxtdir GEOM GEOMDIR BASE UBASE FOLD A_FOLD A_CFBASE B_FOLD B_CFBASE T_FOLD T_CFBASE"
    for var in $vars ; do printf "%30s : %s \n" $var ${!var} ; done 
    source $OPTICKS_HOME/bin/AB_FOLD.sh   # just lists dir content 
fi 

if [ "$arg" == "fold" ]; then 
    echo $FOLD 
fi 



loglevels()
{
    export Dummy=INFO
    export SGeoConfig=INFO
    export SEvt=INFO
    export Ctx=INFO
    #export QSim=INFO
    #export QEvent=INFO 
    export CSGOptiX=INFO
    export G4CXOpticks=INFO 
    export GInstancer=INFO
    export X4PhysicalVolume=INFO
    export U4VolumeMaker=INFO
}
loglevels




if [ "${arg/run}" != "$arg" ]; then 
    echo $msg run $bin
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run $bin error && exit 1 
fi 

if [ "${arg/dbg}" != "$arg" ]; then 
    case $(uname) in
        Linux) gdb_ $bin -ex r  ;;
        Darwin) lldb__ $bin ;; 
    esac
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg $bin error && exit 2 
fi

if [ "${arg/ana}" != "$arg" ]; then 
    export FOLD
    export CFBASE=$T_CFBASE    ## T_CFBASE would seem better otherwise assumes have rerun A with same geom at T (and B)
    export MASK=${MASK:-pos}

    ${IPYTHON:-ipython} --pdb -i $gxtdir/tests/$bin.py     
    [ $? -ne 0 ] && echo $BASH_SOURCE ana $bin error && exit 3 
fi 

if [ "${arg/grab}" != "$arg" ]; then 
    source $gxtdir/../bin/rsync.sh $UBASE 
fi 

if [ "$arg" == "pvcap" -o "$arg" == "pvpub" -o "$arg" == "mpcap" -o "$arg" == "mppub" ]; then
    export CAP_BASE=$FOLD/figs
    export CAP_REL=gxt
    export CAP_STEM=${GEOM}_${OPTICKS_INPUT_PHOTON_LABEL}

    case $arg in 
       pvcap) source pvcap.sh cap  ;;
       mpcap) source mpcap.sh cap  ;;
       pvpub) source pvcap.sh env  ;;
       mppub) source mpcap.sh env  ;;
    esac

    if [ "$arg" == "pvpub" -o "$arg" == "mppub" ]; then 
        source epub.sh 
    fi 
fi 

exit 0 

