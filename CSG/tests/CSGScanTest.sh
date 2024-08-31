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
bin=/tmp/$name
script=CSGScanTest.py 


base=/tmp/$USER/opticks/$name


geom=JustOrb

export CSGSCANTEST_BASE=$base
export CSGSCANTEST_SOLID=$geom

scans="axis rectangle circle"

vars="CSGSCANTEST_BASE CSGSCANTEST_SOLID"


if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/build}" != "$arg" ]; then

    opts=""
    #opts="-DDEBUG=1"    ## very verbose 

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


list-all()
{
    echo $FUNCNAME $*
    local scan
    for scan in $* ; do 
       tmpdir=$base/${scan}_scan
       echo $tmpdir
       ls -l $tmpdir
    done 
}
list-recent(){
   echo $FUNCNAME 
   find $base -newer CSGScanTest.cc -exec ls -l {} \; 
}

if [ "${arg/run}" != "$arg" ]; then 
    for scan in $scans ; do 
        tmpdir=$base/${scan}_scan
        mkdir -p $tmpdir 
    done 
    $bin
    [ $? -ne 0 ] && echo run error && exit 2
    list-recent  
fi 

if [ "${arg/list}" != "$arg" ]; then 
    list-all
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} -i --pdb $script
    [ $? -ne 0 ] && echo ana error && exit 3
fi 

exit 0 
