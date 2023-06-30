#!/bin/bash -l 

name=sdomain_test 
export FOLD=/tmp/$name
mkdir -p $FOLD
bin=$FOLD/$name


gcc $name.cc -std=c++11 -lstdc++ -I.. -o $bin
[ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1

$bin
[ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2

${IPYTHON:-ipython} --pdb -i $name.py 
[ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 3


exit 0 


