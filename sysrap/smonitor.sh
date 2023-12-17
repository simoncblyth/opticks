#!/bin/bash -l 
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

4. make a plot, from the same directory::

    ~/o/sysrap/smonitor.sh ana 


EOU
}

SDIR=$(dirname $(realpath $BASH_SOURCE))

name=smonitor

vars="BASH_SOURCE SDIR PWD name TMP FOLD CUDA_PREFIX CUDA_TARGET bin"

cuda_prefix=/usr/local/cuda-11.7
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}
CUDA_TARGET=$CUDA_PREFIX/targets/x86_64-linux

tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}
export FOLD=$TMP/$name
mkdir -p $FOLD

bin=$FOLD/$name
script=$SDIR/$name.py 

cd $FOLD
LOGDIR=$PWD


defarg="info_build_run_ana"
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

exit 0 

