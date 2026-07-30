#!/bin/bash


cd $(dirname $(realpath $BASH_SOURCE))

name=scie_test
bin=/tmp/$USER/opticks/$name
mkdir -p $(dirname $bin)

gcc $name.cc -std=c++17 -I.. -lstdc++ -lm  -o $bin && $bin


