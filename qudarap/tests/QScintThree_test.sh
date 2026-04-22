#!/bin/bash

usage(){ cat << EOU
QScintThree_test.sh
=====================


::

   ~/o/qudarap/tests/QScintThree_test.sh

    COMP=0 LOGY=1 EDGE=1 BINS=329:601:1 ~/o/qudarap/tests/QScintThree_test.sh pdb
    COMP=1 LOGY=1 EDGE=1 BINS=308:549:1 ~/o/qudarap/tests/QScintThree_test.sh pdb
    COMP=2 LOGY=1 EDGE=1 BINS=329:601:1 ~/o/qudarap/tests/QScintThree_test.sh pdb

    COMP=012 LOGY=1 EDGE=1 BINS=308:601:1 ~/o/qudarap/tests/QScintThree_test.sh pdb


Removing the zero padding avoids the unexpected wavelengths from ICDF lookups::

    QScintThree_test.sh HD:1 COMP:012 BINS:(308, 601, 1) : Chi2/ndf  (LAB  1.0778 (NDF: 269) ) (PPO  0.9357 (NDF: 239) ) (bisMSB  1.1954 (NDF: 269) )
    QScintThree_test.sh HD:1 COMP:012 BINS:(200, 800, 1) : Chi2/ndf  (LAB  1.0815 (NDF: 269) ) (PPO  0.9363 (NDF: 239) ) (bisMSB  1.1989 (NDF: 269) )
    WITH_LERP of the overlap, makes almost no difference - slightly larger Chi2


::

    QScintThree_test.sh HD:1 WITH_LERP:1 COMP:012 BINS:(200, 800, 1) : Chi2/ndf  (LAB  1.0093 (NDF: 269) ) (PPO  0.8849 (NDF: 239) ) (bisMSB  1.0093 (NDF: 269) )
    QScintThree_test.sh HD:1 WITH_LERP:0 COMP:012 BINS:(200, 800, 1) : Chi2/ndf  (LAB  1.0071 (NDF: 269) ) (PPO  0.8847 (NDF: 239) ) (bisMSB  1.0071 (NDF: 269) )


After BinCentered fix::

    QScintThree_test.sh HD:1 WITH_LERP:1 COMP:012 BINS:(200, 800, 1) : Chi2/ndf  (LAB  1.0089 (NDF: 269) ) (PPO  0.8847 (NDF: 239) ) (bisMSB  1.0089 (NDF: 269) ):w
    QScintThree_test.sh HD:1 WITH_LERP:0 COMP:012 BINS:(200, 800, 1) : Chi2/ndf  (LAB  1.0067 (NDF: 269) ) (PPO  0.8847 (NDF: 239) ) (bisMSB  1.0067 (NDF: 269) )


Push the stats to a billion::

    QScintThree_test.sh HD:1 WITH_LERP:0 COMP:012 BINS:(200, 800, 1) : Chi2/ndf  (LAB  1.0002 (NDF: 269) ) (PPO  0.9487 (NDF: 239) ) (bisMSB  1.0002 (NDF: 269) )



Presentation::

    COMP=012 YLIM=1e4,2e7 LOGY=1 EDGE=1 BINS=305:605:0.5 ~/o/qudarap/tests/QScintThree_test.sh pdb



EOU
}

name=QScintThree_test

source $HOME/.opticks/GEOM/GEOM.sh

cd $(dirname $(realpath $BASH_SOURCE))

defarg="info_nvcc_gcc_run_pdb"
arg=${1:-$defarg}

unset Q4ScintThree__SD

if [ -n "$SD" ]; then
    export Q4ScintThree__SD=1
    HD=0
else
    HD=1
fi

if [ -n "$Q4ScintThree__SD" ]; then
   printf " $BASH_SOURCE ===== WARNING HD20 HAS BEEN DISABLED VIA Q4ScintThree__SD \n"
fi



unset QSCINTTHREE_DISABLE_INTERPOLATION
#export QSCINTTHREE_DISABLE_INTERPOLATION=1

if [ -n "$QSCINTTHREE_DISABLE_INTERPOLATION" ]; then
   printf " $BASH_SOURCE ====== WARNING TEXTURE INTERPOLATION DISABLED \n"
fi




tmp=/tmp/$USER/opticks
export TMP=${TMP:-$tmp}
export FOLD=$TMP/${name}_HD$HD
mkdir -p $FOLD

cuo=$FOLD/QScintThree_cu.o
bin=$FOLD/$name
script=$name.py

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}


# ordinary "lo" environment is sufficient
get-cmake-prefix(){ echo $CMAKE_PREFIX_PATH | tr ":" "\n" | grep $1 ; }
CLHEP_PREFIX=$(get-cmake-prefix CLHEP)
GEANT4_PREFIX=$(get-cmake-prefix Geant4)

#spec=M10
#spec=M100
spec=G1

export U4ScintThree__num_wlsamp=$spec
export Q4ScintThree__num_wlsamp=$spec



vars="BASH_SOURCE PWD name defarg arg tmp TMP FOLD cuo bin GEOM CUDA_PREFIX CLHEP_PREFIX GEANT4_PREFIX U4ScintThree__num_wlsamp Q4ScintThree__num_wlsamp"

if [ "${arg/info}" != "$arg" ]; then
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done
fi

if [ "${arg/du}" != "$arg" ]; then
   du -h $FOLD/*
   du -h $FOLD/U4ScintThree/*
fi



if [ "${arg/nvcc}" != "$arg" ]; then
   echo nvcc
   #    -DWITH_LERP \
   nvcc \
       -c \
       ../QScintThree.cu \
       -I.. \
       -I../../sysrap \
       -std=c++17 -lstdc++  \
       -o $cuo
   [ $? -ne 0 ] && echo $BASH_SOURCE : nvcc error && exit 1
fi

if [ "${arg/gcc}" != "$arg" ]; then
   echo gcc
   gcc \
       $name.cc \
       $cuo \
       -std=c++17 -lstdc++ -lcudart -g \
       -DWITH_CUDA \
       -DRNG_PHILOX \
       -I.. \
       -I../../sysrap \
       -I../../u4 \
       -I$CUDA_PREFIX/include \
       -I$CLHEP_PREFIX/include \
       -I$GEANT4_PREFIX/include/Geant4  \
       -L$GEANT4_PREFIX/lib64 \
       -L$CLHEP_PREFIX/lib \
       -lG4global \
       -lG4geometry \
       -lCLHEP \
       -L$CUDA_PREFIX/lib64 \
       -o $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE : gcc error && exit 1
fi

if [ "${arg/run}" != "$arg" ]; then
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2
fi

if [ "${arg/dbg}" != "$arg" ]; then
   source dbg__.sh
   dbg__ $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE : dbg error && exit 2
fi

if [ "${arg/pdb}" != "$arg" ]; then
   mode=-2
   MODE=${MODE:-$mode} ${IPYTHON:-ipython} --pdb -i $script
   [ $? -ne 0 ] && echo $BASH_SOURCE : pdb error && exit 3
fi

exit 0

