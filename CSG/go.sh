#!/bin/bash -l


./build.sh 
[ $? -ne 0 ] && exit 2

./run.sh 
[ $? -ne 0 ] && exit 3

exit 0 

