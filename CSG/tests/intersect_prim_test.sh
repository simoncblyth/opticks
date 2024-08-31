#!/bin/bash 
usage(){ cat << EOU
intersect_prim_test.sh 
=============================

::

    ~/opticks/CSG/tests/intersect_prim_test.sh




Despite not being used presence of SCU.h in SysRap is requiring 
linking with cudart ? 

    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: /tmp/ccjA3IEx.o: in function `void SCU::DownloadVec<float>(std::vector<float, std::allocator<float> >&, float const*, unsigned int)':
    intersect_prim_test.cc:(.text._ZN3SCU11DownloadVecIfEEvRSt6vectorIT_SaIS2_EEPKS2_j[_ZN3SCU11DownloadVecIfEEvRSt6vectorIT_SaIS2_EEPKS2_j]+0x7d): undefined reference to `cudaMemcpy'
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: intersect_prim_test.cc:(.text._ZN3SCU11DownloadVecIfEEvRSt6vectorIT_SaIS2_EEPKS2_j[_ZN3SCU11DownloadVecIfEEvRSt6vectorIT_SaIS2_EEPKS2_j]+0xd8): undefined reference to `cudaGetErrorString'

Coming in via SCSGPrimSpec.h ? But thats not in any .cc ?  



EOU
}

cd $(dirname $BASH_SOURCE)
name=intersect_prim_test

export FOLD=/tmp/$name
mkdir -p $FOLD
bin=$FOLD/$name
script=$name.py 

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

defarg="info_build_run"
arg=${1:-$defarg}

vars="BASH_SOURCE name FOLD bin CUDA_PREFIX arg script"

if [ "${arg/info}" != "$arg" ]; then 
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi

if [ "${arg/build}" != "$arg" ]; then 
    gcc $name.cc \
       ../CSGNode.cc \
       -I..  \
       -std=c++11 -lstdc++ -lm \
       -I${OPTICKS_PREFIX}/externals/plog/include \
       -I${OPTICKS_PREFIX}/include/OKConf \
       -I${OPTICKS_PREFIX}/include/SysRap \
       -L${OPTICKS_PREFIX}/lib64 \
       -lOKConf -lSysRap \
       -L${CUDA_PREFIX}/lib64 -lcudart \
       -I${CUDA_PREFIX}/include \
       -o $bin

    [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1 
fi

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2
fi

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 3
fi

exit 0 

