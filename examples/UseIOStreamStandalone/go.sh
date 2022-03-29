#!/bin/bash -l 

iwd=$PWD
msg="=== $BASH_SOURCE :"
name=UseIOStreamStandalone
tmpdir=/tmp/$USER/opticks/$name
mkdir -p $tmpdir
cd $tmpdir
pwd

cat << EOS > $name.cc
#include <iostream>
int main(int argc, char** argv)
{
    std::cout << "Hello from : " << argv[0] << std::endl ; 
    return 0 ; 
}
EOS

gcc $name.cc -std=c++11 -lstdc++ -o $name 
[ $? -ne 0 ] && echo $msg compilation error && exit 1

./$name
[ $? -ne 0 ] && echo $msg run error && exit 2

exit 0 

