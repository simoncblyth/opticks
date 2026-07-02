#!/bin/bash

usage(){ cat << EOU

~/o/sysrap/tests/SEvt_AddEnvMeta_Test.sh

EOU
}



name=SEvt_AddEnvMeta_Test

bin=$name
script=$name.py


export COMMANDLINE="Check metadata saving COMMANDLINE with spaces in it"
$bin

export FOLD=/tmp/$USER/opticks/$name


${IPYTHON:-ipython} --pdb -i $script




