#!/bin/bash -l 

name=schrono_test 

gcc $name.cc -std=c++11 -lstdc++ -I.. -o /tmp/$name && /tmp/$name




