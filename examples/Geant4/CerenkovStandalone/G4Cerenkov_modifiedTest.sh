#!/bin/bash -l 
usage(){ cat << EOU
~/o/examples/Geant4/CerenkovStandalone/G4Cerenkov_modifiedTest.sh
===================================================================


EOU
}


cd $(dirname $(realpath $BASH_SOURCE))

msg="=== $BASH_SOURCE :"


source $HOME/.opticks/GEOM/GEOM.sh
source cks.bash
cks-env

srcs=(G4Cerenkov_modifiedTest.cc G4Cerenkov_modified.cc OpticksDebug.cc OpticksRandom.cc OpticksUtil.cc)
name=${srcs[0]}
name=${name/.cc}

mkdir -p /tmp/$name


echo $msg srcs : ${srcs[@]} name : $name

if [ -n "$SCAN" ]; then 
    docompile=0
    interactive=0
else
    docompile=1
    interactive=1
fi 


if [ $docompile -eq 1 ]; then
    cks-compile ${srcs[@]}
    eval $(cks-compile ${srcs[@]})
    [ $? -ne 0 ] && echo cks-compile FAIL && exit 1 
fi 


export OPTICKS_RANDOM_SEQPATH=$HOME/.opticks/precooked/QSimTest/rng_sequence/rng_sequence_f_ni1000000_nj16_nk16_tranche100000

cks-run $name 
eval $(cks-run $name) $*
[ $? -ne 0 ] && echo run FAIL && exit 2
echo cks-run succeeds


analysis=0
if [ $analysis -eq 1 ]; then 
    if [ $interactive -eq 1 ]; then 
        ipython -i $name.py
        [ $? -ne 0 ] && echo analysis FAIL && exit 3
    else
        python $name.py
        [ $? -ne 0 ] && echo analysis FAIL && exit 3
    fi
fi 



