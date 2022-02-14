#!/bin/bash -l 

name=signbitTest 

gcc $name.cc -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name
