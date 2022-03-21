#!/bin/bash -l 

source cks.bash
cks-env

msg="=== $BASH_SOURCE :"

srcs=(RINDEXTest.cc OpticksDebug.cc OpticksUtil.cc)
for src in ${srcs[@]} ; do echo $src ; done

name=${srcs[0]}
name=${name/.cc}

echo $msg srcs : ${srcs[@]} name : $name


docompile=1
if [ $docompile -eq 1 ]; then
    cks-compile ${srcs[@]}
    eval $(cks-compile ${srcs[@]})
    [ $? -ne 0 ] && echo compile FAIL && exit 1 
fi 

cks-run $name 
eval $(cks-run $name) $*
[ $? -ne 0 ] && echo run FAIL && exit 2
echo run succeeds


