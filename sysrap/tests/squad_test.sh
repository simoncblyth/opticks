#!/bin/bash -l 

name=squad_test

gcc $name.cc -I.. -I${CUDA_INCLUDEDIR:-/usr/local/cuda/include} -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name 



