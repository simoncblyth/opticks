#!/bin/bash -l 

name=STTFTest 
gcc -g $name.cc -I.. -std=c++11 -lstdc++ -o /tmp/$name 
[ $? -ne 0 ] && echo compile fail && exit 1 

path=/tmp/$USER/opticks/sysrap/tests/STTFTest.jpg 
mkdir -p $(dirname $path)
rm -f $path 

text="$0 : the quick brown fox jumps over the lazy dog 0.123456789" 
export OPTICKS_STTF_PATH=/Library/Fonts/Arial.ttf 
/tmp/$name $path "$text" 
[ $? -ne 0 ] && echo run fail && exit 2

[ ! -f "$path" ] && echo failed to create path $path && exit 3
open $path    

exit 0  

