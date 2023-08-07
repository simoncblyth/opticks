#!/bin/bash -l 
usage(){ cat << EOU

::

    D=1.00000100  ~/opticks/sysrap/tests/float_double_test.sh 
    D=1.48426314  ~/opticks/sysrap/tests/float_double_test.sh 

EOU
}

SDIR=$(cd $(dirname $BASH_SOURCE) && pwd )

name=float_double_test 
bin=/tmp/$name

[ ! -f $bin ] && gcc $SDIR/$name.cc -I$SDIR/.. -std=c++11 -lstdc++ -o $bin 

$bin

