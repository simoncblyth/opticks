#!/bin/bash -l 
usage(){ cat << EOU
sboundary_test.sh
===================

::

    N=160 POLSCALE=10 AOI=BREWSTER ./sboundary_test.sh 
    N=160 POLSCALE=10 AOI=45 ./sboundary_test.sh 

    N=4 POLSCALE=10 AOI=BREWSTER  ./sboundary_test.sh 

EOU
}

name=sboundary_test
export FOLD=/tmp/$name

force=R  # R/T/N
export FORCE=${FORCE:-$force}


mkdir -p $FOLD

defarg=build_run_ana
arg=${1:-$defarg}

if [ "${arg/build}" != "$arg" ]; then 
   gcc $name.cc \
          -std=c++11 -lstdc++ \
          -I.. \
          -DMOCK_CURAND \
          -I/usr/local/cuda/include \
          -I$OPTICKS_PREFIX/externals/glm/glm \
          -I$OPTICKS_PREFIX/externals/plog/include \
          -o $FOLD/$name 

   [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then 

   $FOLD/$name
   [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2 
fi 

if [ "${arg/ana}" != "$arg" ]; then 

   export AFOLD=$FOLD
   export BFOLD=/tmp/qsim_test

   ${IPYTHON:-ipython} --pdb -i $name.py 
   [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 3 
fi 

exit 0 

