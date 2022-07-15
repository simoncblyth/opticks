#!/bin/bash 
usage(){ cat << EOU
AB_FOLD.sh
============

EOU
}

vars="BASH_SOURCE A_FOLD B_FOLD"
for var in $vars ; do printf "%30s : %s \n" "$var" "${!var}" ; done 

echo A_FOLD $A_FOLD
du -hs $A_FOLD
du -hs $A_FOLD/*.npy

echo B_FOLD $B_FOLD
du -hs $B_FOLD
du -hs $B_FOLD/*.npy



