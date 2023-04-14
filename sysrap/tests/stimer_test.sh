#!/bin/bash -l 

name=stimer_test

export FOLD=/tmp/$name
export TTPATH=$FOLD/tt.npy

mkdir -p $FOLD 
bin=$FOLD/$name

gcc $name.cc -std=c++11 -lstdc++ -I.. -o $bin && $bin

${IPYTHON:-ipython} --pdb -i $name.py 





