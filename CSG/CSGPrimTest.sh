#!/bin/bash -l 

#source $OPTICKS_HOME/bin/geocache_hookup.sh 

${IPYTHON:-ipython} -i --pdb -- tests/CSGPrimTest.py 

