#!/bin/bash
usage(){ cat << EOU
sdigest_duplicate_test.sh
===============================

~/o/sysrap/tests/sdigest_duplicate_test/sdigest_duplicate_test.sh

::

    2025-10-20 17:33:04.365 [load $HITFOLD/hit.npy
    2025-10-20 17:34:15.443 ]load $HITFOLD/hit.npy
    2025-10-20 17:34:15.443 hit (997737665, 4, 4, ) num_items 997737665 num_threads[NPROC] 56 chunk_size 17816745 num_chunks 56
    2025-10-20 17:34:15.443 i   0 start          0 end   17816745
    2025-10-20 17:34:15.443 i   1 start   17816745 end   35633490
    2025-10-20 17:34:15.443 i   2 start   35633490 end   53450235
    2025-10-20 17:34:15.443 i   3 start   53450235 end   71266980
    ...
    2025-10-20 17:34:54.691  join 
    2025-10-20 17:34:54.691  join 
    2025-10-20 17:34:54.691  join 
    2025-10-20 17:34:54.691  join 
    2025-10-20 17:34:54.691 merge 
    2025-10-20 17:55:54.061Seen 997737665Found 0 unique duplicated 4x4 hits.
    (ok) A[blyth@localhost sdigest_duplicate_test]$ 


HMM: 22 minutes to look for duplicates amongst a billion hits

* see CUDA thrust approach in ~/j/hit_digest


EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=sdigest_duplicate_test

tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}

FOLD=$TMP/$name
mkdir -p $FOLD

nproc_default=$(nproc)
export NPROC=${NPROC:-$nproc_default}

bin=$FOLD/$name

opt="-Wdeprecated-declarations"
case $(uname) in
  Darwin) opt="" ;;
   Linux) opt="-lssl -lcrypto " ;;
esac


#hitfold=/data1/blyth/tmp/GEOM/J25_4_0_opticks_Debug/CSGOptiXSMTest/ALL1_Debug_Philox_vvvlarge_evt/A000
hitfold=/data1/blyth/tmp/GEOM/J25_4_0_opticks_Debug/CSGOptiXSMTest/ALL1_Debug_Philox_vvvvvlarge_evt/A000
export HITFOLD=$hitfold

defarg=info_build_run
arg=${1:-$defarg}

vars="BASH_SOURCE defarg arg name bin FOLD HITFOLD nproc_default NPROC"

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%20s : %s\n" "$var" "${!var}" ; done
fi

if [ "${arg/build}" != "$arg" ]; then
   gcc $name.cc -std=c++17 -g -Wall -lstdc++ $opt -I../.. -o $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1
fi

if [ "${arg/run}" != "$arg" ]; then
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
fi


exit 0


