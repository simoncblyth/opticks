#!/bin/bash 
usage(){ cat << EOU
log.sh 
=======

Parse and present logfile delta timings::

   LOG=/path/to/log.log ~/opticks/bin/log.sh 
   LOG=/tmp/G4CXSimtraceTest.log ~/opticks/bin/log.sh 

EOU
}

log=/tmp/G4CXSimtraceTest.log
export LOG=${LOG:-$log}
${IPYTHON:-ipython} --pdb -i $(dirname $BASH_SOURCE)/log.py  


