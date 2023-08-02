#!/bin/bash -l 

name=sstr_test ; gcc $name.cc -g -std=c++11 -lstdc++ -I.. -o /tmp/$name && /tmp/$name

