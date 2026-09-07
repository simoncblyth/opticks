#!/usr/bin/env bash
usage(){ cat << EOU
~/o/sysrap/tests/vector_clear_test.sh
======================================

With gcc11, no crash with Debug OR Release options::

    [lo] A[blyth@localhost tests]$ ~/o/sysrap/tests/vector_clear_test.sh
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/contrib/gcc/11.4.1/bin/gcc
    /tmp/vector_clear_test_Release -- Compiled with gcc 11.4.1 20231218
     all.size 4 num 4 [ 0 0 0 0 ] -- before deliberate clear bug
     all.size 0 num 4 [ 100 200 300 400 ] -- after clear and undefined [] filling
    /tmp/vector_clear_test_Debug -- Compiled with gcc 11.4.1 20231218
     all.size 4 num 4 [ 0 0 0 0 ] -- before deliberate clear bug
     all.size 0 num 4 [ 100 200 300 400 ] -- after clear and undefined [] filling

With gcc15, Release build does not crash, Debug build does::

    [lo] A[blyth@localhost tests]$ ~/o/sysrap/tests/vector_clear_test.sh
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/contrib/gcc/15.1.0/bin/gcc
    /tmp/vector_clear_test_Release -- Compiled with gcc 15.1.0
     all.size 4 num 4 [ 0 0 0 0 ] -- before deliberate clear bug
     all.size 0 num 4 [ 100 200 300 400 ] -- after clear and undefined [] filling
    /tmp/vector_clear_test_Debug -- Compiled with gcc 15.1.0
     all.size 4 num 4 [ 0 0 0 0 ] -- before deliberate clear bug
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/contrib/gcc/15.1.0/include/c++/15.1.0/bits/stl_vector.h:1263: std::vector<_Tp, _Alloc>::reference std::vector<_Tp, _Alloc>::operator[](size_type) [with _Tp = int; _Alloc = std::allocator<int>; reference = int&; size_type = long unsigned int]: Assertion '__n < this->size()' failed.
    /home/blyth/o/sysrap/tests/vector_clear_test.sh: line 50: 1451600 Aborted                 (core dumped) $bin_Debug
    /home/blyth/o/sysrap/tests/vector_clear_test.sh 51 - ERROR FROM /tmp/vector_clear_test_Debug





EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=vector_clear_test

bin_Release=/tmp/${name}_Release
bin_Debug=/tmp/${name}_Debug

opt_Release="-O3 -DNDEBUG"
opt_Debug="-g -O0"

which gcc
#gcc --version

gcc $name.cc -std=c++17 $opt_Release -lstdc++ -o $bin_Release
[ $? -ne 0 ] && echo $BASH_SOURCE $LINENO - gcc error && exit 1

gcc $name.cc -std=c++17 $opt_Debug    -lstdc++ -o $bin_Debug
[ $? -ne 0 ] && echo $BASH_SOURCE $LINENO - gcc error && exit 1

$bin_Release
[ $? -ne 0 ] && echo $BASH_SOURCE $LINENO - ERROR FROM $bin_Release

$bin_Debug
[ $? -ne 0 ] && echo $BASH_SOURCE $LINENO - ERROR FROM $bin_Debug



