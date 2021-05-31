#!/bin/bash -l 

dtag=-1
tag=${1:-$dtag}

cat="g4live"
det="det"
src="natural"
pfx="G4OKTest"

export OPTICKS_ANA_DEFAULTS="src=$src,cat=$cat,det=$det,tag=$tag,pfx=$pfx"

ipython -i --pdb $(which evt.py) 

