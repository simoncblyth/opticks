#!/bin/bash -l 
usage(){ cat << EOU
UseGLMQuat.sh
===============

::
 
   ~/o/examples/UseGLMQuat/UseGLMQuat.sh 

EOU
}


path=$(realpath $BASH_SOURCE)
name=$(basename $path)
stem=${name/.sh}

bin=/tmp/$stem

defarg="info_build_run"
arg=${1:-$defarg}

cd $(dirname $path)

vars="BASH_SOURCE path stem bin PWD"

if [ "${arg/info}" != "$arg" ]; then 
   for var in $vars ; do printf "%20s : %s\n" "$var" "${!var}" ; done 
fi 

if [ "${arg/build}" != "$arg" ]; then 
    glm-
    gcc $stem.cc -std=c++11 -lstdc++ -I$(glm-prefix)  -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1 
fi

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2 
fi

exit 0 







