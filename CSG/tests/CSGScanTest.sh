#!/bin/bash 
usage(){ cat << EOU
CSGScanTest.sh
===============

::

    ~/o/CSG/tests/CSGScanTest.sh

EOU
}


cd $(dirname $(realpath $BASH_SOURCE))

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}   # just use some CUDA headers, not using GPU 

defarg="info_build_run_ana"
arg=${1:-$defarg}


name=CSGScanTest 
fold=/tmp/$USER/opticks/$name
bdir=/tmp/$USER/opticks/${name}.build
mkdir -p $bdir

bin=$bdir/$name
script=CSGScanTest.py 


export FOLD=$fold

geom=JustOrb
export GEOM=$geom
export BASE=$FOLD/$GEOM

vars="FOLD GEOM BASE"

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 



if [ "${arg/build}" != "$arg" ]; then

    opts=""
    #opts="-DDEBUG=1"    ## very verbose 

    cui=../CSGScan.cu
    cuo=$bdir/CSGScan_cu.o
    nvcc -c $cui \
             -std=c++11 -lstdc++ \
             -I.. \
             -I$OPTICKS_PREFIX/include/SysRap \
             -o $cuo 
    [ $? -ne 0 ] && echo $BASH_SOURCE : nvcc compile error cui $cui cuo $cuo  && exit 1

    srcs="$name.cc 
          ../CSGFoundry.cc 
          ../CSGImport.cc 
          ../CSGSolid.cc 
          ../CSGCopy.cc 
          ../CSGPrim.cc 
          ../CSGNode.cc 
          ../CSGScan.cc 
          ../CSGTarget.cc 
          ../CSGMaker.cc 
          ../CU.cc 
          $cuo
          "

    gcc \
        $srcs \
        -I.. \
        -std=c++11 -lm \
        $opts \
        -I${CUDA_PREFIX}/include \
        -I${OPTICKS_PREFIX}/externals/glm/glm \
        -I${OPTICKS_PREFIX}/include/SysRap \
        -I${OPTICKS_PREFIX}/externals/plog/include \
        -L${CUDA_PREFIX}/lib64 -lcudart -lstdc++ \
        -L${OPTICKS_PREFIX}/lib64 \
        -lSysRap \
        -DWITH_CHILD \
        -DWITH_VERBOSE \
        -o $bin

    [ $? -ne 0 ] && echo build error && exit 1

fi 







list-recent(){
   echo $FUNCNAME 
   find $FOLD -newer CSGScanTest.cc -exec ls -l {} \; 
}

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo run error && exit 2
    list-recent  
fi 

if [ "${arg/grab}" != "$arg" ]; then 
    source ../../bin/BASE_grab.sh $arg
    [ $? -ne 0 ] && echo grab error && exit 4
fi 


if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} -i --pdb $script
    [ $? -ne 0 ] && echo ana error && exit 4
fi 

exit 0 
