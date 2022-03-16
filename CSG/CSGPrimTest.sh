#!/bin/bash -l 
usage(){ cat << EOU
CSGPrimTest.sh
=================

::
 
    PY=1 ./CSGPrimTest.sh 
         use python script rather thah default binary 

    ELV=103 ./CSGPrimTest.sh 
         ELV SBitSet prim selection based on meshIdx with CSGCopy::Select  
         first character t means NOT (tilde)


The argument is used by opticks-switch-key to set the OPTICKS_KEY selecting 
the geometry to use. 

old
    some old reference geometry 
new
    recent addition
remote
    grabbed CSGFoundry 
asis
    OPTICKS_KEY 
last
    latest development version  

EOU
}

default=remote
arg=${1:-$default}

opticks-switch-key $arg 

if [ -n "$PY" ]; then 
     ${IPYTHON:-ipython} -i --pdb -- tests/CSGPrimTest.py 
else
    bin=CSGPrimTest 

    if [ -n "$DEBUG" ]; then 
       if [ "$(uname)" == "Darwin" ]; then
           lldb__ $bin
       else
           gdb $bin
       fi
    else
       $bin
    fi 
fi



