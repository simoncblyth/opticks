#!/bin/bash -l 

cat="g4live"
det="det"
src="natural"
pfx="source"
tag=${1:-1}

export OPTICKS_ANA_DEFAULTS="src=$src,cat=$cat,det=$det,tag=$tag,pfx=$pfx"

ipython -i --pdb $(which evt.py) 

