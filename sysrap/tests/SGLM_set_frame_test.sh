#!/bin/bash -l 

msg="=== $BASH_SOURCE :"
name=SGLM_set_frame_test 
bin=/tmp/$name

# directory to load sframe.npy from 
export BASE=/tmp/blyth/opticks/GEOM/V0J008/CSGOptiXRdrTest
defarg="build_run_info_cat"
arg=${1:-$defarg}

#DESCNAME=SGLM__writeDesc.log 
DESCNAME=SGLM_set_frame_test.log

DESCPATH=$BASE/$DESCNAME
vars="name bin BASE arg DESCNAME DESCPATH"


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

if [ "${arg/cat}" != "$arg" ]; then
    cat $DESCPATH
    ls -l $DESCPATH
fi 

if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 


exit 0 



