#!/bin/bash -l

./build.sh
[ $? -ne 0 ] && echo $0 build error && exit 1 

./run.sh $*
[ $? -ne 0 ] && echo $0 run error && exit 2

exit 0 
