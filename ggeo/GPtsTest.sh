#!/bin/bash -l
usage(){ cat << EOU
GPtsTest.sh
=================

The arguments are passed to geocache_hookup to select the geometry to use. 

old
    some old reference geometry 
new
    recent addition under testing 
last
    latest development version  

EOU
}

msg="=== $BASH_SOURCE :"

source $OPTICKS_HOME/bin/geocache_hookup.sh ${1:-new}

${IPYTHON:-ipython} -i --pdb --  GPtsTest.py


