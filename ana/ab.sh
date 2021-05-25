#!/bin/bash -l 

usage(){ cat << EOU
ab.sh
======

::

   ab.sh 1 --nocompare   # early stage debugging 

   ab.sh 1 
   ab.sh 2



EOU
}


cat="g4live"
det="det"
src="natural"
pfx="source"

tag=${1:-1}
shift 

export OPTICKS_ANA_DEFAULTS="src=$src,cat=$cat,det=$det,tag=$tag,pfx=$pfx"

#script=evt.py
script=ab.py 
#script=profile_.py 

ipython -i --pdb $(which $script) -- $*


