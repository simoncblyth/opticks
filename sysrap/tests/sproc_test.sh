#!/bin/bash -l 

name=sproc_test 
bin=${TMP:-/tmp/$USER/opticks}/$name
mkdir -p $(dirname $bin)

gcc $name.cc -I.. -std=c++11 -lstdc++ -lm -o $bin && $bin


