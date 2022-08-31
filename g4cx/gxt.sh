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

ana imp::

    vi tests/G4CXSimtraceTest.py ../ana/simtrace_plot.py ../ana/pvplt.py ../ana/feature.py 



As B uses A and T uses A+B the running order is:

A. gx ; ./gxs.sh 
B. u4 ; ./u4s.sh 
T. gx ; ./gxt.sh 


analog timings : showing log lines taking more than 2 percent of total time
-------------------------------------------------------------------------------

::

    In [1]: log[2]
    Out[1]: 
                     timestamp :        DTS-prev :        DFS-frst :msg
    2022-08-23 23:46:16.288000 :      0.3370[34] :      0.3370[34] : INFO  [57430] [main@20] ] cu first 
    2022-08-23 23:46:16.504000 :      0.1780[18] :      0.5530[56] : INFO  [57430] [QSim::UploadComponents@111] ] new QRng 
    2022-08-23 23:46:16.613000 :      0.0800[ 8] :      0.6620[67] : INFO  [57430] [CSGOptiX::initCtx@322] ]
    2022-08-23 23:46:16.640000 :      0.0270[ 3] :      0.6890[69] : INFO  [57430] [CSGOptiX::initPIP@333] ]
    2022-08-23 23:46:16.740000 :      0.0360[ 4] :      0.7890[80] : INFO  [57430] [CSGFoundry::getFrame@2880]  fr sframe::desc inst 0 frs -1
    2022-08-23 23:46:16.910000 :      0.1140[11] :      0.9590[97] : INFO  [57430] [SEvt::gather@1378]  k        simtrace a  <f4(627000, 4, 4, )
    2022-08-23 23:46:16.942000 :      0.0320[ 3] :      0.9910[100] : INFO  [57430] [SEvt::save@1505] ] fold.save 


                             - :                 :                 :G4CXSimtraceTest.log
    2022-08-23 23:46:15.951000 :                 :                 :start
    2022-08-23 23:46:16.943000 :                 :                 :end
                             - :                 :      0.9920[100] :total seconds
                             - :                 :      2.0000[100] :pc_cut

::

    CUDA_VISIBLE_DEVICES=0,1 ./gxt.sh       ## appears to give same timings as default 
    CUDA_VISIBLE_DEVICES=0 ./gxt.sh         ## slightly reduced CUDA latency from 0.31s down to 0.25s
    CUDA_VISIBLE_DEVICES=1 ./gxt.sh         ## slightly reduced CUDA latency from 0.31s down to 0.23s


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
  analog)  QUIET=1 ;;
esac

bin=G4CXSimtraceTest
log=$bin.log
source $(dirname $BASH_SOURCE)/../bin/COMMON.sh 

if [ -n "$OPTICKS_INPUT_PHOTON" ]; then 
   unset OPTICKS_INPUT_PHOTON  ## simtrace running and input photons cannot be used together 
fi 


UGEOMDIR=${GEOMDIR//$HOME\/}

BASE=$GEOMDIR/$bin
UBASE=${BASE//$HOME\/}    # UBASE relative to HOME to handle rsync between different HOME
FOLD=$BASE/ALL            # corresponds SEvt::save() with SEvt::SetReldir("ALL")

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
    vars="BASH_SOURCE arg bin defarg GEOM GEOMDIR UGEOMDIR BASE UBASE FOLD A_FOLD A_CFBASE B_FOLD B_CFBASE T_FOLD T_CFBASE J001_GDMLPath"
    for var in $vars ; do printf "%30s : %s \n" $var ${!var} ; done 
    source $OPTICKS_HOME/bin/AB_FOLD.sh   # just lists dir content 
fi 

if [ "$arg" == "fold" ]; then 
    echo $FOLD 
fi 


cehigh()
{
    : increase genstep resolution, see sysrap/tests/SFrameGenstep_MakeCenterExtentGensteps_Test.sh
    export CEHIGH_0=-11:-9:0:0:-2:0:1000:4
    export CEHIGH_1=9:11:0:0:-2:0:1000:4
    export CEHIGH_2=-1:1:0:0:-2:0:1000:4
}
cehigh


loglevels()
{
    export Dummy=INFO
    #export SGeoConfig=INFO
    export SEventConfig=INFO
    #export SEvt=INFO          # lots of AddGenstep output, messing with timings : actually log times not changed much, but it feels slow over network
    #export Ctx=INFO
    export QSim=INFO
    export QBase=INFO
    export SSim=INFO
    export SBT=INFO
    export IAS_Builder=INFO
    #export QEvent=INFO 
    export CSGOptiX=INFO
    export G4CXOpticks=INFO 
    export CSGFoundry=INFO
    #export GInstancer=INFO
    #export X4PhysicalVolume=INFO
    #export U4VolumeMaker=INFO
}
loglevels




if [ "${arg/run}" != "$arg" ]; then 
    echo $msg run $bin log $log 
    if [ -f "$log" ]; then 
       rm $log 
    fi 

    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run $bin error && exit 1 
fi 

if [ "grablog" == "$arg" ]; then
    scp P:opticks/g4cx/$log .
fi 

if [ "analog" == "$arg" ]; then 
    echo $msg analog log $log 
    if [ -f "$log" ]; then 
        LOG=$log $(dirname $BASH_SOURCE)/../bin/log.sh 
    fi 
fi 

if [ "${arg/dbg}" != "$arg" ]; then 
    case $(uname) in
        Linux) gdb_ $bin -ex r  ;;
        Darwin) lldb__ $bin ;; 
    esac
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg $bin error && exit 2 
fi

if [ "ana" == "$arg" ]; then 
    export FOLD
    export CFBASE=$T_CFBASE    ## T_CFBASE would seem better otherwise assumes have rerun A with same geom at T (and B)
    export MASK=${MASK:-pos}
    export TOPLINE="gxt.sh/$bin.py : GEOM $GEOM " 

    ${IPYTHON:-ipython} --pdb -i $(dirname $BASH_SOURCE)/tests/$bin.py     
    [ $? -ne 0 ] && echo $BASH_SOURCE ana $bin error && exit 3 
fi 

if [ "anachk" == "$arg" ]; then 
    export FOLD
    export CFBASE=$T_CFBASE    ## T_CFBASE would seem better otherwise assumes have rerun A with same geom at T (and B)
    export MASK=${MASK:-pos}

    ${IPYTHON:-ipython} --pdb -i $(dirname $BASH_SOURCE)/tests/CSGFoundryLoadTest.py     
    [ $? -ne 0 ] && echo $BASH_SOURCE anachk $bin error && exit 3 
fi 




if [ "grab" == "$arg" ]; then 
    source $(dirname $BASH_SOURCE)/../bin/rsync.sh $UGEOMDIR
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

