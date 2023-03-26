#!/bin/bash -l 

bin=SEvt_AddEnvMeta_Test

export COMMANDLINE="Check metadata saving COMMANDLINE with spaces in it"
$bin

export FOLD=/tmp/$USER/opticks/$bin
${IPYTHON:-ipython} --pdb -i $bin.py 




