#!/bin/bash -l 
usage(){ cat << EOU
U4Polycone_test.sh
===================

::

   # -L$OPTICKS_PREFIX/lib \
   # -lSysRap \

EOU
}


SDIR=$(cd $(dirname $BASH_SOURCE) && pwd)

name=U4Polycone_test

export FOLD=/tmp/$name
mkdir -p $FOLD

bin=$FOLD/$name
script=$SDIR/$name.py 

CUDA_PREFIX=/usr/local/cuda

clhep-
g4-

defarg="build_run"
arg=${1:-$defarg}

#opt="-DWITH_SND"
opt=""


export sn__level=2
export s_pool_level=2


vars="BASH_SOURCE arg SDIR FOLD bin script opt"

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done
fi

if [ "${arg/build}" != "$arg" ]; then
         
    gcc \
         $opt \
         $SDIR/$name.cc \
         $SDIR/../../sysrap/sn.cc \
         $SDIR/../../sysrap/s_pa.cc \
         $SDIR/../../sysrap/s_bb.cc \
         $SDIR/../../sysrap/s_tv.cc \
         $SDIR/../../sysrap/s_csg.cc \
         \
         $SDIR/../../sysrap/snd.cc \
         $SDIR/../../sysrap/scsg.cc \
         -I$SDIR/.. \
         -std=c++11 -lstdc++ \
         -I$HOME/opticks/sysrap \
         -I$CUDA_PREFIX/include \
         -I$OPTICKS_PREFIX/externals/glm/glm \
         -I$(clhep-prefix)/include \
         -I$(g4-prefix)/include/Geant4  \
         -L$(g4-prefix)/lib \
         -L$(clhep-prefix)/lib \
           -lG4global \
           -lG4geometry \
           -lCLHEP \
          -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
fi 

if [ "${arg/ana}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 3
fi 

exit 0


