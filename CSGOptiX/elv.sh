#!/bin/bash 
usage(){ cat << EOU
elv.sh / emm.sh :  analysis of ELV/EMM scan metadata
======================================================

NB emm.sh is a symbolic link to elv.sh so this single script
handles both with mode switched based on the scriptstem being "emm" or "elv"

Usage::

   ~/o/CSGOptiX/emm.sh jpg
   ~/o/CSGOptiX/elv.sh jpg
         ## write list of jpg paths in ascending render time order

   ~/o/CSGOptiX/emm.sh txt
   ~/o/CSGOptiX/elv.sh txt
         ## write TXT table with ordered render times 

   ~/o/CSGOptiX/emm.sh rst
   ~/o/CSGOptiX/elv.sh rst
         ## write RST table with ordered render times 


Call stack::

    elv.sh OR emm.sh 
    ./cxr_overview.sh jstab
    source $SDIR/../bin/BASE_grab.sh $arg
    PYTHONPATH=../.. ${IPYTHON:-ipython} --pdb  ../ana/snap.py --  --globptn "$globptn" --refjpgpfx "$refjpgpfx" $SNAP_ARGS


Check out the Panel geometry::

    EMM=9, ~/o/cx.sh 


EOU
}


defarg="txt"
arg=${1:-$defarg}

if [ "$arg" == "ALL" ]; then
    types="jpg txt rst"
else
    types=$arg
fi 


cd $(dirname $(realpath $BASH_SOURCE))
scriptname=$(basename $BASH_SOURCE)   ## NB not with realpath as want to see name of symbolic link not the real path 
scriptstem=${scriptname/.sh}

MODE=$scriptstem

SCAN=scan-$MODE
LIM=512


case $MODE in 
  elv) estem=cxr_view ;;
  emm) estem=cxr_overview ;;
esac


vars="0 BASH_SOURCE scriptname scriptstem MODE SCAN LIM arg defarg types vars"
for var in $vars ; do printf "%20s :  %s \n" "$var" "${!var}" ; done 


for typ in $types 
do 
   outpath=/tmp/${MODE}_${typ}.txt

   export SNAP_estem=$estem

   export SNAP_outpath=$outpath
   export SNAP_selectmode=$MODE
   export SNAP_selectspec=all 
   export SNAP_dump=$typ
   export SNAP_LIMIT=$LIM  

   if [ "$MODE" == "elv" ]; then
       export SNAP_globptn='$BASE/cxr_view*elv*.jpg' 
       export SNAP_refjpgpfx="/env/presentation/cxr/cxr_view"
   elif [ "$MODE" == "emm" ]; then
       export SNAP_globptn='$BASE/cxr_overview*emm*.jpg' 
       export SNAP_refjpgpfx="/env/presentation/cxr/cxr_overview"
   fi

   SCAN=$SCAN  ./$estem.sh jstab    ## this runs snap.py 

   wc -l $outpath
   cat $outpath 
   wc -l $outpath
done


