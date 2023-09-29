#!/bin/bash -l 

name=stamp_test 

export FOLD=/tmp/$USER/$name
export IDENTITY="$BASH_SOURCE $(uname -n) $(date)"
mkdir -p $FOLD
bin=$FOLD/$name

#defarg="build_run_ana"
defarg="info_build_run"
arg=${1:-$defarg}

REMOTE=lxslc708.ihep.ac.cn

np_base=..
NP_BASE=${NP_BASE:-$np_base}

vars="BASH_SOURCE arg bin FOLD REMOTE"


if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/build}" != "$arg" ]; then 
    gcc $name.cc -std=c++11 -lstdc++ -lm -I$NP_BASE/np -I.. -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run  error && exit 2
fi 

if [ "${arg/grab}" != "$arg" ]; then 
    rsync -zarv --progress  \
          --include="*.npy" \
          --include="*.txt" \
          --include="*.jpg" \
          --include="*.png" \
          "$REMOTE:$FOLD/" "$FOLD"

    [ $? -ne 0 ] && echo $BASH_SOURCE grab error && exit 3
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $name.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ana  error && exit 5
fi 

if [ "${arg/ls}" != "$arg" ]; then 
    find "$FOLD" -type f
    [ $? -ne 0 ] && echo $BASH_SOURCE ls error && exit 5
fi 


exit 0 

