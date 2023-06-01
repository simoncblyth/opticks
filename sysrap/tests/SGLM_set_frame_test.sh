#!/bin/bash -l 

msg="=== $BASH_SOURCE :"
name=SGLM_set_frame_test 
bin=/tmp/$name

# directory to load sframe.npy from 
export BASE=/tmp/blyth/opticks/GEOM/V0J008/CSGOptiXRdrTest

defarg="build_run"
arg=${1:-$defarg}

if [ "${arg/build}" != "$arg" ]; then 
    gcc $name.cc -g -Wall -std=c++11 -lstdc++ -I.. \
             -I$OPTICKS_PREFIX/externals/glm/glm \
             -I/usr/local/cuda/include -o $bin
    [ $? -ne 0 ] && echo $msg build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $msg run error && exit 2 
fi 

exit 0 


