#!/bin/bash 
usage(){ cat << EOU
smonitor.sh : NVML based GPU memory monitor
=============================================

Usage:

1. start the monitor process in one tab::
 
    ~/o/sysrap/smonitor.sh   # builds and starts runloop 

2. start the GPU memory using job in another tab

3. once the job has completed, ctrl-C the monitor process
   which catches the SIGINT signal and saves the memory 
   profile into smonitor.npy array  

4. make a plot::

    ~/o/sysrap/smonitor.sh grab
    ~/o/sysrap/smonitor.sh ana 


5. screen capture the plot with annotation 

    ~/o/sysrap/smonitor.sh mpcap 
    PUB=cxs_min_igs_with_rg_dummy ~/o/sysrap/smonitor.sh mppub 


EOU
}

SDIR=$(dirname $(realpath $BASH_SOURCE))

name=smonitor

vars="BASH_SOURCE SDIR PWD name TMP FOLD CUDA_PREFIX CUDA_TARGET bin"

cuda_prefix=/usr/local/cuda-11.7
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}
CUDA_TARGET=$CUDA_PREFIX/targets/x86_64-linux

tmp=/tmp/$USER/opticksGG
TMP=${TMP:-$tmp}
export FOLD=$TMP/$name
mkdir -p $FOLD

bin=$FOLD/$name
script=$SDIR/$name.py 

cd $FOLD
LOGDIR=$PWD

#stem=smonitor_okjob
stem=smonitor_cxs_min_igs
pub=no-PUB
export STEM=${STEM:-$stem}
export PUB=${PUB:-$pub}

export smonitor__SLEEP_US=100000  # 0.1s 


defarg="info_build_run"
arg=${1:-$defarg}

if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%25s : %s \n" "$var" "${!var}" ; done
fi 

if [ "${arg/build}" != "$arg" ]; then 
    gcc $SDIR/$name.cc -std=c++11 -lstdc++ \
         -I$SDIR \
         -I$CUDA_TARGET/include \
         -L$CUDA_TARGET/lib \
         -lnvidia-ml  \
         -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2 
fi 

if [ "${arg/grab}" != "$arg" ]; then 
    source $OPTICKS_HOME/bin/rsync.sh $LOGDIR
    [ $? -ne 0 ] && echo $BASH_SOURCE grab error && exit 3 
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $script 
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 4
fi 

if [ "$arg" == "mpcap" -o "$arg" == "mppub" ]; then
    export CAP_BASE=$FOLD/figs
    export CAP_REL=smonitor
    export CAP_STEM=${STEM}_${PUB}
    case $arg in  
       mpcap) source mpcap.sh cap  ;;  
       mppub) source mpcap.sh env  ;;  
    esac
    if [ "$arg" == "mppub" ]; then 
        source epub.sh 
    fi  
fi 

exit 0
