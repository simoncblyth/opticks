#!/bin/bash -l 
usage(){ cat << EOU
STTFTest.sh
=============

::

   ~/o/sysrap/tests/STTFTest.sh 


Creates img, annotates and saves to file testing STTF.h and SIMG.h 

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=STTFTest 
bin=${TMP:-/tmp/$USER/opticks}/$name/$name
mkdir -p $(dirname $bin)

gcc $name.cc \
     -g -std=c++11 -lstdc++ -lm \
     -I$OPTICKS_PREFIX/include/SysRap \
      -o $bin
[ $? -ne 0 ] && echo compile fail && exit 1 

path=${TMP:-/tmp/$USER/opticks}/sysrap/tests/STTFTest.jpg 
mkdir -p $(dirname $path)
rm -f $path 

$bin $path 
[ $? -ne 0 ] && echo run fail && exit 2

[ ! -f "$path" ] && echo failed to create path $path && exit 3
open $path    

exit 0  

