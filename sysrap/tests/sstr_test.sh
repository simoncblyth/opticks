#!/bin/bash -l 
usage(){ cat << EOU

::

   ~/opticks/sysrap/tests/sstr_test.sh 


EOU
}


name=sstr_test 
bin=${TMP:-/tmp/$USER/opticks}/$name
mkdir -p $(dirname $bin)

cd $(dirname $(realpath $BASH_SOURCE)) 
gcc $name.cc -g -std=c++11 -lstdc++ -I.. -o $bin && $bin


