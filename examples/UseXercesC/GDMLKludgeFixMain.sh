#!/bin/bash -l

name=GDMLKludgeFixMain
bin=/tmp/$name 
srcs="$name.cc GDMLKludgeFix.cc GDMLRead.cc GDMLWrite.cc"

build=0
if [ ! -f "$bin" ]; then
    echo binary $bin does not exist  
    build=1
else
    for src in $srcs ; do 
       if [ $src -nt $bin ]; then 
          echo binary $bin exists but the src $src has been modified
          build=1 
       fi 
    done    
fi

if [ $build -eq 1 ]; then 
    xercesc- 
    gcc $srcs \
          -std=c++11 \
          -I$(xercesc-prefix)/include \
          -L$(xercesc-prefix)/lib -lxerces-c \
          -lstdc++ \
          -o $bin
    [ $? -ne 0 ] && echo compile error && exit 1 
else
    echo using preexisting binary $bin
fi


srcdefault=$HOME/origin2.gdml
src=${1:-$srcdefault}
dst=${src/.gdml}_GDMLKludgeFix.gdml

cmd="/tmp/$name $src"
echo $cmd
eval $cmd
[ $? -ne 0 ] && echo run error && exit 2


cmd="diff $src $dst"
echo $cmd
eval $cmd

exit 0 

