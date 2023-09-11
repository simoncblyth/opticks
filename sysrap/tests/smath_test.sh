#/bin/bash -l 

name=smath_test 

cd $(dirname $BASH_SOURCE)

FOLD=/tmp/$name
mkdir -p $FOLD
bin=$FOLD/$name
export FOLD

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}
opt=-DMOCK_CUDA

defarg="info_build_run_ana"
arg=${1:-$defarg}
vars="BASH_SOURCE name arg FOLD bin opt"

if [ "${arg/info}" != "$arg" ]; then 
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi



if [ "${arg/build}" != "$arg" ]; then 
    gcc $name.cc $opt -std=c++11 -lstdc++ -I.. -I$CUDA_PREFIX/include -o $bin 
    [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2 
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    echo ${IPYTHON:-ipython} --pdb -i $name.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 3 
fi 

exit 0 

