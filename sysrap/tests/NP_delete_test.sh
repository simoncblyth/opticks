#!/bin/bash -l 
usage(){ cat << EOU
NP_delete_test.sh
===================

~/opticks/sysrap/tests/NP_delete_test.sh 

* either CLEAR or DELETE or both all look the same wrt RSS effects. 
* with neither CLEAR or DELETE get monotonic growth 
* seems RSS ends up at not the largest allocation but the second largest, 
  it does not go back down to the initial value 

CONCLUSION : SHOULD NOT WORRY TOO MUCH ABOUT RSS VALUE (THE SYSTEM IS 
MANAGING MEMORY ALLOCATIONS ACCORDING TO ITS ALGORITHMS) 
JUST MAKE SURE RSS IS NOT ALWAYS GOING UP : AVOID FIREHOSE "LEAK" 
BY NOT OMITTING TO DELETE LARGE ARRAYS. 

EOU
}

name=NP_delete_test 

TMP=${TMP:-/tmp/$USER/opticks}
export FOLD=$TMP/$name
mkdir -p $FOLD
bin=$FOLD/$name

cd $(dirname $BASH_SOURCE)

defarg="build_run_cat_ana"
arg=${1:-$defarg}

#export CLEAR=1
export DELETE=1 
export TEST=t2

if [ "${arg/build}" != "$arg" ]; then 
    gcc $name.cc -std=c++11 -lstdc++ -I.. -o $bin 
    [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1 
fi

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2
fi 

if [ "${arg/cat}" != "$arg" ]; then 
    cat $FOLD/run_meta.txt
    [ $? -ne 0 ] && echo $BASH_SOURCE : cat error && exit 3
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $name.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 3
fi 

exit 0


