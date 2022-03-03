#!/bin/bash -l 

usage(){ cat << EOU
rundbg.sh
===========

EOU
}

./run.sh --savegparts --earlyexit $*


