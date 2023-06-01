#!/bin/bash -l 
usage(){ cat << EOU
SGLM_set_frame_test.sh
========================

This standalone test aims to duplicate the sframe.h/SGLM.h view calculations 
done by CSGOptiX::RenderMain. In more detail, the SGLM_set_frame_test.cc does:

1. loads $BASE/sframe.npy into sframe.h instance

   * the sframe is persisted by CSGOptiX::render_snap

2. instanciates SGLM.h and invokes SGLM::set_frame with the loaded frame
3. invokes SGLM::writeDesc writing the SGLM::desc to $BASE/SGLM_set_frame_test.log

::
   
    st
    ./SGLM_set_frame_test.sh build_run_info_cat_diff  
    ./SGLM_set_frame_test.sh build
    ./SGLM_set_frame_test.sh run
    ./SGLM_set_frame_test.sh info
    ./SGLM_set_frame_test.sh cat
    ./SGLM_set_frame_test.sh diff
         # compare the SGLM::desc logged from CSGOptiX::render_snap and SGLM_set_frame_test.cc 

EOU
}


defarg="build_run_info_cat_diff"
arg=${1:-$defarg}

msg="=== $BASH_SOURCE :"
name=SGLM_set_frame_test 
bin=/tmp/$name

# directory to load sframe.npy from 
base=/tmp/$USER/opticks/GEOM/V0J008/CSGOptiXRdrTest
export BASE=${BASE:-$base}

tmin=0.5
export TMIN=${TMIN:-$tmin}


DESC_REF=CSGOptiX__render_snap.log
DESC_NAME=SGLM_set_frame_test.log
DESC_PATH=$BASE/$DESC_NAME
vars="name bin BASE arg DESC_NAME DESC_PATH TMIN"


if [ "${arg/build}" != "$arg" ]; then 
    gcc $name.cc -g -Wall -std=c++11 -lstdc++ -I.. \
             -I$OPTICKS_PREFIX/externals/glm/glm \
             -I/usr/local/cuda/include -o $bin
    [ $? -ne 0 ] && echo $msg build error && exit 1 
    echo $BASH_SOURCE : build OK
fi 

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $msg run error && exit 2 
    echo $BASH_SOURCE : run OK
fi 

if [ "${arg/cat}" != "$arg" ]; then
    cat $DESC_PATH
    ls -l $DESC_PATH
    echo $BASH_SOURCE : cat 
fi 

if [ "${arg/diff}" != "$arg" ]; then 
    cmd="diff $BASE/$DESC_REF $BASE/$DESC_NAME"
    echo $BASH_SOURCE diff : $cmd
    eval $cmd
fi 

if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
    echo $BASH_SOURCE : info
fi 

exit 0 

