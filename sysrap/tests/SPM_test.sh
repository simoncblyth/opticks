#!/bin/bash
usage(){ cat << EOU
SPM_test.sh
===========

::

    ~/o/sysrap/tests/SPM_test.sh

    OPTICKS_MERGE_WINDOW=50    ~/o/sysrap/tests/SPM_test.sh info_run
    OPTICKS_MERGE_WINDOW=10000 ~/o/sysrap/tests/SPM_test.sh info_run

Trend correct, but not monotonically downwards, "window edge" precision effect perhaps::

    (ok) A[blyth@localhost sysrap]$ ~/o/sysrap/tests/SPM_test.sh  scan
    SPM_test::merge_partial_select_async num_photonlite 9195 select_flagmask 8192 merge_window_ns     1 hitlitemerged (1611, 4, ) merge_result.count 1611
    SPM_test::merge_partial_select_async num_photonlite 9195 select_flagmask 8192 merge_window_ns     5 hitlitemerged (1606, 4, ) merge_result.count 1606
    SPM_test::merge_partial_select_async num_photonlite 9195 select_flagmask 8192 merge_window_ns    10 hitlitemerged (1602, 4, ) merge_result.count 1602
    SPM_test::merge_partial_select_async num_photonlite 9195 select_flagmask 8192 merge_window_ns    20 hitlitemerged (1588, 4, ) merge_result.count 1588
    SPM_test::merge_partial_select_async num_photonlite 9195 select_flagmask 8192 merge_window_ns    30 hitlitemerged (1584, 4, ) merge_result.count 1584
    SPM_test::merge_partial_select_async num_photonlite 9195 select_flagmask 8192 merge_window_ns    40 hitlitemerged (1582, 4, ) merge_result.count 1582
    SPM_test::merge_partial_select_async num_photonlite 9195 select_flagmask 8192 merge_window_ns    50 hitlitemerged (1577, 4, ) merge_result.count 1577
    SPM_test::merge_partial_select_async num_photonlite 9195 select_flagmask 8192 merge_window_ns    60 hitlitemerged (1580, 4, ) merge_result.count 1580
    SPM_test::merge_partial_select_async num_photonlite 9195 select_flagmask 8192 merge_window_ns    70 hitlitemerged (1573, 4, ) merge_result.count 1573
    SPM_test::merge_partial_select_async num_photonlite 9195 select_flagmask 8192 merge_window_ns    80 hitlitemerged (1567, 4, ) merge_result.count 1567
    SPM_test::merge_partial_select_async num_photonlite 9195 select_flagmask 8192 merge_window_ns    90 hitlitemerged (1563, 4, ) merge_result.count 1563
    SPM_test::merge_partial_select_async num_photonlite 9195 select_flagmask 8192 merge_window_ns   100 hitlitemerged (1566, 4, ) merge_result.count 1566
    SPM_test::merge_partial_select_async num_photonlite 9195 select_flagmask 8192 merge_window_ns   200 hitlitemerged (1561, 4, ) merge_result.count 1561
    SPM_test::merge_partial_select_async num_photonlite 9195 select_flagmask 8192 merge_window_ns   300 hitlitemerged (1552, 4, ) merge_result.count 1552
    SPM_test::merge_partial_select_async num_photonlite 9195 select_flagmask 8192 merge_window_ns   400 hitlitemerged (1547, 4, ) merge_result.count 1547
    SPM_test::merge_partial_select_async num_photonlite 9195 select_flagmask 8192 merge_window_ns   500 hitlitemerged (1545, 4, ) merge_result.count 1545
    SPM_test::merge_partial_select_async num_photonlite 9195 select_flagmask 8192 merge_window_ns  1000 hitlitemerged (1544, 4, ) merge_result.count 1544
    (ok) A[blyth@localhost sysrap]$


EOU
}

cd $(dirname $(realpath $BASH_SOURCE))
name=SPM_test
script=$name.py

tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}
FOLD=$TMP/$name
bin=$FOLD/$name

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

afold=/tmp/blyth/opticks/GEOM/J25_4_0_opticks_Debug/python3.11/ALL0_no_opticks_event_name/A000
export AFOLD=${AFOLD:-$afold}

#test=merge_partial_select
test=merge_partial_select_async
#test=dump_partial

export TEST=${TEST:-$test}

opticks_merge_window=1
export OPTICKS_MERGE_WINDOW=${OPTICKS_MERGE_WINDOW:-$opticks_merge_window}


mkdir -p $FOLD

defarg="info_nvcc_gcc_run"
arg=${1:-$defarg}

vv="BASH_SOURCE name tmp TMP FOLD bin defarg arg cuda_prefix CUDA_PREFIX test TEST AFOLD OPTICKS_MERGE_WINDOW"

if [ "${arg/info}" != "$arg" ]; then
    for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done
fi

if [ "${arg/ls}" != "$arg" ]; then
   ff="AFOLD FOLD"
   for f in $ff ; do
      cmd="ls -alst ${!f} # $f"
      echo $cmd
      eval $cmd
   done
fi


if [ "${arg/nvcc}" != "$arg" ]; then
   nvcc -c ../SU.cu -I.. -std=c++17 --extended-lambda  -o $FOLD/SU.o
   [ $? -ne 0 ] && echo $FUNCNAME nvcc compile error && exit 1
fi

if [ "${arg/nvcc}" != "$arg" ]; then
   nvcc -c ../SPM.cu -I.. -std=c++17 --extended-lambda  -o $FOLD/SPM.o
   [ $? -ne 0 ] && echo $FUNCNAME nvcc compile error && exit 1
fi

if [ "${arg/gcc}" != "$arg" ]; then
   gcc $name.cc $FOLD/SU.o $FOLD/SPM.o -std=c++17 -lstdc++ -I.. -I$CUDA_PREFIX/include -L$CUDA_PREFIX/lib64 -lcudart -lm  -o $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE gcc compile error && exit 2
fi

if [ "${arg/run}" != "$arg" ]; then
   pushd $FOLD > /dev/null
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 3
   popd > /dev/null
fi

if [ "${arg/scan}" != "$arg" ]; then
   pushd $FOLD > /dev/null

   tt="1 5 10 20 30 40 50 60 70 80 90 100 200 300 400 500 1000"
   for t in $tt ; do
      OPTICKS_MERGE_WINDOW=$t $bin
   done

   [ $? -ne 0 ] && echo $BASH_SOURCE scan error && exit 3
   popd > /dev/null
fi





if [ "${arg/dbg}" != "$arg" ]; then
   source dbg__.sh
   pushd $FOLD
   dbg__ $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 4
   popd
fi

if [ "${arg/pdb}" != "$arg" ]; then
   export AFOLD=/data1/blyth/tmp/GEOM/J25_4_0_opticks_Debug/CSGOptiXSMTest/ALL1_Debug_Philox_merge/A000
   ${IPYTHON:-ipython} --pdb -i $script
   [ $? -ne 0 ] && echo $BASH_SOURCE pdb error && exit 5
fi

exit 0

