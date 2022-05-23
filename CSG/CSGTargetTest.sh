#!/bin/bash -l 

usage(){ cat << EOU
CSGTargetTest.sh 
===================

::
    
    c ; METH=descInstance IDX=37684 ./CSGTargetTest.sh remote   


EOU
}


default=remote
arg=${1:-$default}

opticks-switch-key $arg 

CSGTargetTest 



