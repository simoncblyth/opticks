#!/bin/bash
usage(){ cat << EOU
csg_intersect_prim_test.sh
=============================

Purely CPU side testing of headers used with CUDA GPU side.

::

    ~/o/CSG/tests/csg_intersect_prim_test.sh

    MODE=3 CIRCLE=0,0,50,100 NCIRCLE=0,0,1 ~/o/CSG/tests/csg_intersect_prim_test.sh pdb


EOU
}

name=csg_intersect_prim_test
cd $(dirname $(realpath $BASH_SOURCE))

#test=HalfSpaceOne
#test=HalfCylinderOne
test=HalfCylinderXY

export TEST=${TEST:-$test}

TMP=${TMP:-/tmp/$USER/opticks}
export FOLD=$TMP/$name
mkdir -p $FOLD
bin=$FOLD/$name
script=$name.py

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

defarg="info_build_run_pdb"
arg=${1:-$defarg}

vars="BASH_SOURCE PWD TEST name TMP FOLD bin CUDA_PREFIX arg script"

if [ "${arg/info}" != "$arg" ]; then
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done
fi

if [ "${arg/build}" != "$arg" ]; then
    gcc \
       $name.cc \
       ../CSGNode.cc \
       -std=c++17 -lstdc++ -lm \
       -I..  \
       -I${OPTICKS_PREFIX}/externals/plog/include \
       -I${OPTICKS_PREFIX}/include/OKConf \
       -I${OPTICKS_PREFIX}/include/SysRap \
       -L${OPTICKS_PREFIX}/lib64 \
       -lOKConf -lSysRap \
       -L${CUDA_PREFIX}/lib64 \
       -lcudart \
       -I${CUDA_PREFIX}/include \
       -DDEBUG_HALFSPACE \
       -o $bin

    [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1
fi

if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2
fi

if [ "${arg/pdb}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $BASH_SOURCE : pdb error && exit 3
fi

if [ "${arg/ana}" != "$arg" ]; then
    ${PYTHON:-python} $script
    [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 4
fi

exit 0

