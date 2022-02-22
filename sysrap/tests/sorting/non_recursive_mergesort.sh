#!/bin/bash -l 

name=non_recursive_mergesort 

gcc $name.cc -g -std=c++11 -I$OPTICKS_PREFIX/include/SysRap -lstdc++ -o /tmp/$name && /tmp/$name
