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
          -fvisibility=hidden \
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

else
    echo using preexisting binary $bin
fi


notes_about_logging(){ cat << EON

Note that the "-fvisibility=hidden" is necessary for PLOG logging level control to work. 
Without it get all logging being emitted see sysrap/PLOG_review.rst

EON
}



srcdefault=$OPTICKS_PREFIX/origin_11dec2021.gdml

msg="=== $BASH_SOURCE :"

src=${1:-$srcdefault}
dst=${src/.gdml}_CGDMLKludge.gdml   # this name change is hardcoded

echo $msg src $src
echo $msg dst $dst


[ ! -f "$src" ] && echo $msg src ERROR $src does not exist && exit 1 

cmd="$bin $src"
echo $cmd
eval $cmd
[ $? -ne 0 ] && echo run error && exit 3

cmd="diff $src $dst"
echo $cmd
#eval $cmd

exit 0 

