#!/bin/bash -l

msg="=== $BASH_SOURCE :"

./build.sh
[ $? -ne 0 ] && echo $msg : build error && exit 1

./run.sh $*
[ $? -ne 0 ] && echo $msg : run error && exit 2

exit 0 

