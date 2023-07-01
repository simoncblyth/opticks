#!/bin/bash -l 

name=sidxname_test 

gcc $name.cc -I.. -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name
