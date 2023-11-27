#!/bin/bash -l 

name=SEvt__addGenstep_test 

cd $(dirname $BASH_SOURCE) 

TMP=${TMP:-/tmp/$USER/opticks}
export FOLD=$TMP/$name
mkdir -p $FOLD
bin=$FOLD/$name

CUDA_PREFIX=/usr/local/cuda

gcc $name.cc -std=c++11 -lstdc++ -I.. -I$CUDA_PREFIX/include -o $bin && $bin && ${IPYTHON:-ipython} --pdb -i $name.py

[ $? -ne 0 ] && echo $BASH_SOURCE ERROR && exit 1
exit 0 

