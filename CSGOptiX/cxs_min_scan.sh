#!/bin/bash -l 
usage(){ cat << EOU
cxs_min_scan.sh
=================

This does single event runs with OPTICKS_MAX_BOUNCE varied 
from from 0 to 31 with corresponding OPTICKS_START_INDEX 
such that the actually separate runs are all written 
into a common "run" folder without overwriting event folds. 

The run_meta.txt is however overwritten, so used "meta"
command to copy that into the event folder. But thats in 
the wrong place so would need special casing to use that 
metadata. 

How to create plots of launch time vs MAX_BOUNCE 
-------------------------------------------------------

::

   ~/opticks/CSGOptiX/cxs_min_scan.sh 
   vi ~/opticks/notes/issues/OPTICKS_MAX_BOUNCE_scanning.rst

Workstation::

    ~/opticks/cxs_min_scan.sh  ## using symbolic link 
    
Laptop::

    ~/opticks/cxs_min.sh grab 
    PLOT=Substamp_ONE_maxb_scan PICK=A ~/opticks/sreport.sh 
    PLOT=Substamp_ONE_maxb_scan PICK=A ~/opticks/sreport.sh mpcap
    PLOT=Substamp_ONE_maxb_scan PICK=A PUB=expensive_tail ~/opticks/sreport.sh mppub

EOU
}

SOURCE=$([ -L $BASH_SOURCE ] && readlink $BASH_SOURCE || echo $BASH_SOURCE)
SDIR=$(cd $(dirname $SOURCE) && pwd)
script=$SDIR/cxs_min.sh

export OPTICKS_SCANNER=$SOURCE
export OPTICKS_RUNNING_MODE=SRM_TORCH
export OPTICKS_NUM_EVENT=1
export OPTICKS_NUM_PHOTON=H1

ii=$(seq 0 31)
for i in $ii ; do
   echo $BASH_SOURCE : i $i 
   export OPTICKS_MAX_BOUNCE=$i 
   export OPTICKS_START_INDEX=$i
   export OPTICKS_SCAN_INDEX=A$(printf "%0.3d" $i) 
   $script run_meta
   [ $? -ne 0 ] && echo $BASH_SOURCE : ERROR RUNNING SCRIPT $script && exit 1 
done

exit 0 

