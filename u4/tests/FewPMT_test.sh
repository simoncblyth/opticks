#!/bin/bash -l 

source FewPMT.sh 
name=FewPMT_test
gcc $name.cc -I$HOME/np -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name



