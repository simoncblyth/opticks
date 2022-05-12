#!/bin/bash -l 

source cks.bash
cks-env

msg="=== $BASH_SOURCE :"

srcs=(OpticksRandomTest.cc OpticksRandom.cc OpticksUtil.cc)
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


seqdir="/tmp/$USER/opticks/QSimTest/rng_sequence/rng_sequence_f_ni1000000_nj16_nk16_tranche100000"
#export OPTICKS_RANDOM_SEQPATH=$seqdir
export OPTICKS_RANDOM_SEQPATH=$seqdir/rng_sequence_f_ni100000_nj16_nk16_ioffset000000.npy 


cks-run $name 
eval $(cks-run $name) $*
[ $? -ne 0 ] && echo run FAIL && exit 2
echo run succeeds


