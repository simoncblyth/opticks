#!/bin/bash -l 
usage(){ cat << EOU
cxs_min_scan.sh
=================

~/opticks/CSGOptiX/cxs_min_scan.sh 

EOU
}


SOURCE=$([ -L $BASH_SOURCE ] && readlink $BASH_SOURCE || echo $BASH_SOURCE)
SDIR=$(cd $(dirname $SOURCE) && pwd)
script=$SDIR/cxs_min.sh

ii=$(seq 0 31)
for i in $ii ; do
   echo $BASH_SOURCE : i $i 
   export OPTICKS_MAX_BOUNCE=$i 
   export OPTICKS_START_INDEX=$i
   $script
   [ $? -ne 0 ] && echo $BASH_SOURCE : ERROR RUNNING SCRIPT $script && exit 1 
done

exit 0 

