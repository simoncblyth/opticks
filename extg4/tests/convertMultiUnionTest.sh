#!/bin/bash -l 

bin=convertMultiUnionTest


export X4Solid=INFO

$bin
[ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 1

exit 0 

