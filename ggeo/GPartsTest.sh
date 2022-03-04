#!/bin/bash -l

usage(){ cat << EOU
GPartsTest.sh
==============

EOU
}


msg="=== $BASH_SOURCE :"

source $OPTICKS_HOME/bin/geocache_hookup.sh

${IPYTHON:-ipython} -i --pdb --  GPartsTest.py



