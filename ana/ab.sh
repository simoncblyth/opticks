#!/bin/bash -l 

cat="g4live"
det="det"
src="natural"
pfx="source"
tag=${1:-1}

export OPTICKS_ANA_DEFAULTS="src=$src,cat=$cat,det=$det,tag=$tag,pfx=$pfx"

#script=evt.py
script=ab.py 
#script=profile_.py 

ipython -i --pdb $(which $script) 


