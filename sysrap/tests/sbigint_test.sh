#!/bin/bash -l 
usage(){ cat << EOU
sbigint_test.sh
===================

A 32 step photon history can be represented
by an 128 bit integer.  Unfortunately 64 bit
is the largest size int thats in the C++ standard. 

What operations are needed:

* count unique values in large array of int128 
* find indices of occurrences of a value in the array  



:google:`std::vector of __int128`

https://codereview.stackexchange.com/questions/220809/128-bit-unsigned-integer

HMM: dealing with __int128 doesnt look easy


https://chromium.googlesource.com/external/github.com/abseil/abseil-cpp/+/HEAD/absl/numeric/int128.h

https://forums.swift.org/t/128-bit-operations-in-swift/21639/13

https://github.com/Jitsusama/UInt128



EOU
}


name=sbigint_test

TMP=${TMP:-/tmp/$USER/opticks}
export FOLD=$TMP/$name

mkdir -p $FOLD
bin=$FOLD/$name

gcc $name.cc -std=c++11 -lstdc++ -o $bin && $bin



