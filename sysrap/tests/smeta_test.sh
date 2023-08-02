#!/bin/bash -l 

name=smeta_test 
bin=/tmp/$name

defarg="info_build_run"
arg=${1:-$defarg}
vars="BASH_SOURCE name arg GEOM"

export V1J009_GEOMList="red,green,blue"

source $HOME/.opticks/GEOM/GEOM.sh 

if [ "${arg/info}" != "$arg" ]; then
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/build}" != "$arg" ]; then
   gcc $name.cc -I.. -std=c++11 -lstdc++ -o $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2
fi 

exit 0 

