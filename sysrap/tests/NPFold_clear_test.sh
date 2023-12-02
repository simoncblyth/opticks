#!/bin/bash -l 
usage(){ cat << EOU
NPFold_clear_test.sh
===================

~/opticks/sysrap/tests/NPFold_clear_test.sh 

EOU
}

name=NPFold_clear_test 

TMP=${TMP:-/tmp/$USER/opticks}
export FOLD=$TMP/$name
mkdir -p $FOLD
bin=$FOLD/$name

cd $(dirname $BASH_SOURCE)

defarg="build_run_cat_ana"
arg=${1:-$defarg}

export TEST=t0

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
    [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 4
fi 

exit 0


