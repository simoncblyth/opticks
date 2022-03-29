#!/bin/bash -l 

msg="=== $BASH_SOURCE :"
name=UseIOStreamStandalone
tmpdir=/tmp/$USER/opticks/$name
mkdir -p $tmpdir

gcc $name.cc -std=c++11 -lstdc++ -o $tmpdir/$name 
[ $? -ne 0 ] && echo $msg compilation error && exit 1

$tmpdir/$name
[ $? -ne 0 ] && echo $msg run error && exit 2

exit 0 

