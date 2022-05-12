#!/bin/bash -l 

source cks.bash
cks-env

msg="=== $BASH_SOURCE :"

srcs=(OpticksUtilTest.cc OpticksUtil.cc)
for src in ${srcs[@]} ; do echo $src ; done

name=${srcs[0]/.cc}

echo $msg srcs : ${srcs[@]} name : $name


docompile=1
if [ $docompile -eq 1 ]; then
    cks-compile ${srcs[@]}
    eval $(cks-compile ${srcs[@]})
    [ $? -ne 0 ] && echo compile FAIL && exit 1 
fi 


export OPTICKS_RANDOM_SEQPATH=/tmp/$USER/opticks/QSimTest/rng_sequence/rng_sequence_f_ni1000000_nj16_nk16_tranche100000

cks-run $name 
eval $(cks-run $name) $*
[ $? -ne 0 ] && echo run FAIL && exit 2
echo run succeeds

exit 0
