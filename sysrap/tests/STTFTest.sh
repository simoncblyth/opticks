#!/bin/bash -l 
usage(){ cat << EOU
STTFTest.sh
=============

Build an run STTFTest with arguments to select the path 
to save the image and the text to include within the image. 

EOU
}

opticks-
plog-

name=STTFTest 
bin=${TMP:-/tmp/$USER/opticks}/$name
mkdir -p $(dirname $bin)

gcc -g $name.cc -I.. -I$(plog-prefix)/include -L$(opticks-prefix)/lib -lSysRap -std=c++11 -lstdc++ -o $bin
[ $? -ne 0 ] && echo compile fail && exit 1 

path=${TMP:-/tmp/$USER/opticks}/sysrap/tests/STTFTest.jpg 
mkdir -p $(dirname $path)
rm -f $path 

text="$0 : the quick brown fox jumps over the lazy dog 0.123456789" 

#export OPTICKS_STTF_PATH=/Library/Fonts/Arial.ttf 

echo OPTICKS_STTF_PATH $OPTICKS_STTF_PATH


$bin $path "$text" 
[ $? -ne 0 ] && echo run fail && exit 2

[ ! -f "$path" ] && echo failed to create path $path && exit 3
open $path    

exit 0  

