#!/bin/bash -l 

name=strided_rangeTest
bin=/tmp/$name

nvcc $name.cu -I.. -I../../sysrap -o $bin
[ $? -ne 0 ] && echo $BASH_SOURCE  : build error && exit 1

$bin
[ $? -ne 0 ] && echo $BASH_SOURCE  : run error && exit 2

exit 0 



