#!/bin/bash -l 
usage(){ cat << EOU
UseGLMQuat_Arcball.sh
========================

::
 
   ~/o/examples/UseGLMQuat/UseGLMQuat_Arcball.sh 

EOU
}


path=$(realpath $BASH_SOURCE)
name=$(basename $path)
stem=${name/.sh}

bin=/tmp/$stem

cd $(dirname $path)

defarg="info_build_run"
arg=${1:-$defarg}


vars="path stem bin PWD"

if [ "${arg/info}" != "$arg" ]; then 
   for var in $vars ; do printf "%20s : %s\n" "$var" "${!var}" ; done 
fi 

if [ "${arg/build}" != "$arg" ]; then 
    glm-
    gcc $stem.cc -std=c++11 -lstdc++ -I$(glm-prefix) -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1 
fi

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2 
fi

exit 0 







