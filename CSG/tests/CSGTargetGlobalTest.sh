#!/bin/bash -l 

moi=solidXJfixture:64
export MOI=${MOI:-$moi}

${IPYTHON:-ipython} -i CSGTargetGlobalTest.py 


