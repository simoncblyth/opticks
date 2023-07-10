#!/bin/bash -l 


opticks_key_remote_dir=$(opticks-key-remote-dir) 
CGREL=${CGREL:-CSG_GGeo}
xbase=$opticks_key_remote_dir/$CGREL
cfbase=$HOME/$xbase

echo $cfbase

export CFBASE=$cfbase
export FOLD=$CFBASE/CSGFoundry/SSim

${IPYTHON:-ipython} --pdb -i SSim_Test.py 



