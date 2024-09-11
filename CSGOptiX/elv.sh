#!/bin/bash 
usage(){ cat << EOU
elv.sh : analysis of ELV scan metadata
=========================================

::

   ~/o/CSGOptiX/elv.sh 
         ## default ALL does : jpg txt rst

   ~/o/CSGOptiX/elv.sh jpg
         ## write list of jpg paths in ascending render time order

   ~/o/CSGOptiX/elv.sh txt
         ## write TXT table with ordered render times 

   ~/o/CSGOptiX/elv.sh rst
         ## write RST table with ordered render times 


Call stack::

    elv.sh
    ./cxr_overview.sh jstab
    source $SDIR/../bin/BASE_grab.sh $arg
    PYTHONPATH=../.. ${IPYTHON:-ipython} --pdb  ../ana/snap.py --  --globptn "$globptn" --refjpgpfx "$refjpgpfx" $SNAP_ARGS

EOU
}

defarg="ALL"
arg=${1:-$defarg}

if [ "$arg" == "ALL" ]; then
    types="jpg txt rst"
else
    types=$arg
fi 


cd $(dirname $(realpath $BASH_SOURCE))

SCAN=scan-emm
#SCAN=scan-elv
LIM=512

for typ in $types 
do 
   outpath=/tmp/elv_${typ}.txt
   snap_args="--$typ --out --outpath=$outpath --selectmode emm"
   SELECTSPEC=all SCAN=$SCAN SNAP_LIMIT=$LIM SNAP_ARGS="$snap_args" ./cxr_overview.sh jstab
   echo $outpath
   cat $outpath 
done





