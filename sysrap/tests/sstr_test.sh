#!/bin/bash -l 

name=sstr_test 
bin=${TMP:-/tmp/$USER/opticks}/$name
mkdir -p $(dirname $bin)

gcc $name.cc -g -std=c++11 -lstdc++ -I.. -o $bin && $bin

