#!/bin/bash -l 
usage(){ cat << EOU
elv.sh : analysis of ELV scan metadata
=========================================

::

   ./elv.sh ### default ALL does : jpg txt rst

   ./elv.sh jpg
   ./elv.sh txt
   ./elv.sh rst

EOU
}

defarg="ALL"
arg=${1:-$defarg}

if [ "$arg" == "ALL" ]; then
    types="jpg txt rst"
else
    types=$arg
fi 

DIR=$(dirname $BASH_SOURCE)
SCAN=scan-elv
LIM=512

for typ in $types 
do 
   outpath=/tmp/elv_${typ}.txt
   snap_args="--$typ --out --outpath=$outpath"
   CVD=1 SELECTSPEC=all SCAN=$SCAN SNAP_LIMIT=$LIM SNAP_ARGS="$snap_args" $DIR/cxr_overview.sh jstab
   echo $outpath
   cat $outpath 
done





