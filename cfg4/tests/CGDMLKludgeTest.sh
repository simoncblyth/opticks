#!/bin/bash -l

name=CGDMLKludgeTest

main=$name.cc
srcs="../CGDMLKludge.cc ../CGDMLKludgeRead.cc ../CGDMLKludgeWrite.cc"

tmpdir=/tmp/CGDMLKludge
mkdir -p $tmpdir 

lib=CGDMLKludge
bin=$tmpdir/$name


build=0
if [ ! -f "$bin" ]; then
    echo binary $bin does not exist  
    build=1
else
    for src in $main $srcs ; do 
       if [ $src -nt $bin ]; then 
          echo binary $bin exists but the src $src has been modified
          build=1 
       fi 
    done    
fi

if [ $build -eq 1 ]; then 
    xercesc- 

    gcc $srcs \
          -shared \
          -std=c++11 \
          -I.. \
          $(opticks-config --cflags SysRap) \
          -I$(xercesc-prefix)/include \
          -L$(xercesc-prefix)/lib -lxerces-c \
          -lstdc++ \
          $(opticks-config --libs SysRap) \
          -o $tmpdir/lib$lib.dylib

    [ $? -ne 0 ] && echo lib compile error && exit 1 

    gcc $main \
          -std=c++11 \
          -I.. \
          -DOPTICKS_SYSRAP \
          -I/usr/local/opticks/include/SysRap \
          -I/usr/local/opticks/include/OKConf \
          -I/usr/local/opticks/externals/plog/include \
          -I$(xercesc-prefix)/include \
          -L$(xercesc-prefix)/lib -lxerces-c \
          -L/usr/local/opticks/lib -lSysRap \
          -L$tmpdir -l$lib \
          -lstdc++ \
          -o $bin
    [ $? -ne 0 ] && echo main compile error && exit 2

    #      -DOPTICKS_CFG4 \
    # $(opticks-config --cflags SysRap) \
    # $(opticks-config --libs SysRap) \

    # something about this standalone build prevents the logging level control from 
    # working and resulting in all logging being emitted 
    # maybe it needs its own equivalent of CFG4_LOG.hh
    # see sysrap/PLOG_review.rst

else
    echo using preexisting binary $bin
fi


srcdefault=$HOME/origin2.gdml
src=${1:-$srcdefault}
dst=${src/.gdml}_CGDMLKludge.gdml

cmd="$bin $src"
echo $cmd
eval $cmd
[ $? -ne 0 ] && echo run error && exit 3

cmd="diff $src $dst"
echo $cmd
#eval $cmd

exit 0 

