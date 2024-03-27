#!/bin/bash -l 
usage(){ cat << EOU
SGLM_frame_targetting_test.sh
================================

::
   
    ~/o/sysrap/tests/SGLM_frame_targetting_test.sh build_run_info_cat_diff  
    ~/o/sysrap/tests/SGLM_frame_targetting_test.sh build
    ~/o/sysrap/tests/SGLM_frame_targetting_test.sh run
    ~/o/sysrap/tests/SGLM_frame_targetting_test.sh info
    ~/o/sysrap/tests/SGLM_frame_targetting_test.sh cat
    ~/o/sysrap/tests/SGLM_frame_targetting_test.sh diff

EOU
}


defarg="build_run_info_cat_diff"
arg=${1:-$defarg}

br="------------------------------------------------------------------------------"
msg="=== $BASH_SOURCE :"
name=SGLM_frame_targetting_test 

tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}
FOLD=$TMP/$name
bin=$FOLD/$name
diff_cmd="( cd $FOLD && vimdiff A.log B.log)"

mkdir -p $FOLD
export FOLD

cd $(dirname $(realpath $BASH_SOURCE))

tmin=0.5
#eye=1000,1000,1000
#eye=3.7878,3.7878,3.7878
#eye=-1,-1,0
eye=0,-3,0


#escale=asis
escale=extent

export TMIN=${TMIN:-$tmin}
export EYE=${EYE:-$eye}
export ESCALE=${ESCALE:-$escale}


cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}


if [ "${arg/build}" != "$arg" ]; then 
    gcc $name.cc -g -Wall -std=c++11 -lstdc++ -lm  -I.. \
             -I$OPTICKS_PREFIX/externals/glm/glm \
             -I$CUDA_PREFIX/include -o $bin
    [ $? -ne 0 ] && echo $msg build error && exit 1 
    echo $BASH_SOURCE : build OK
fi 

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $msg run error && exit 2 
    echo $BASH_SOURCE : run OK
fi 

if [ "${arg/diff}" != "$arg" ]; then 
    echo $BASH_SOURCE diff 
    echo $cmd
    echo $br
    eval $diff_cmd
    echo $br
fi 

if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
    echo $BASH_SOURCE : info
fi 

exit 0 

