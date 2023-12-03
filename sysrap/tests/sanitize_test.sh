#!/bin/bash -l 
usage(){ cat << EOU
sanitize_test.sh
===================


https://github.com/google/sanitizers/wiki/AddressSanitizer

If you want gdb to stop after asan has reported an error, set a breakpoint on  __sanitizer::Die 


EOU
}



name=sanitize_test
bin=/tmp/$name


gcc $name.cc -fsanitize=address -O1 -fno-omit-frame-pointer -g -std=c++11 -lstdc++ -o $bin 

BP=__sanitizer::Die dbg__ $bin





