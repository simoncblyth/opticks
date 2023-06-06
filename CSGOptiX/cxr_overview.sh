#!/bin/bash -l 
usage(){ cat << EOU
cxr_overview.sh
================

::

   GDB=gdb    ./cxr_overview.sh 
   GDB=lldb__ ./cxr_overview.sh 

   EMM=0, ./cxr_overview.sh 
   EMM=1, ./cxr_overview.sh 

   CVD=1 ./cxr_overview.sh jstab

   CVD=1 SELECTSPEC=all SNAP_LIMIT=512 ./cxr_overview.sh jstab 

   CVD=1 SELECTSPEC=all SNAP_LIMIT=512 SNAP_ARGS="--jpg --out --outpath=/tmp/time_ordered_jpg.txt" ./cxr_overview.sh jstab
   CVD=1 SELECTSPEC=all SNAP_LIMIT=512 SNAP_ARGS="--txt --out --outpath=/tmp/time_ordered_txt.txt" ./cxr_overview.sh jstab

   open -n $(cat /tmp/time_ordered_jpg.txt)   ## open all jpg in time order 
   vim -R /tmp/time_ordered_txt.txt           ## view txt table to see the ELV exclusion names

   epsilon:~ blyth$ cp /tmp/time_ordered_txt.txt ~/j/issues/cxr_scan_cxr_overview_time_ordered_txt.txt
   epsilon:~ blyth$ cp /tmp/time_ordered_jpg.txt ~/j/issues/cxr_scan_cxr_overview_time_ordered_jpg.txt





* cxr_overview.sh render times with JUNO 

JUNO (trunk:Dec, 2021)
   without   0.0054 

JUNO (trunk:Mar 2, 2022)

   without  0.0126
   WITH_PRD 0.0143
   without  0.0126
   without  0.0125  
   WITH_PRD 0.0143   (here and above WITH_PRD used attribs and payload values at 8 without reason)
   WITH_PRD 0.0125   (now with attribs and payload values reduced to 2)
   WITH_PRD 0.0124  

   WITH_PRD not-WITH_CONTIGUOUS 0.0123


EOU
}

DIR=$(dirname $BASH_SOURCE)

case $(uname) in 
  Linux) defarg="run" ;;
  Darwin) defarg="grab_open" ;; 
esac
arg=${1:-$defarg}

source $HOME/.opticks/GEOM/GEOM.sh  # exports GEOM envvar selecting geometry 

escale=extent
moi=-1
tmin=0.4
eye=-0.6,0,0,1
icam=0
zoom=1.5


export ESCALE=${ESCALE:-$escale}
export MOI=${MOI:-$moi} 
export TMIN=${TMIN:-$tmin} 
export EYE=${EYE:-$eye}
export ICAM=${ICAM:-$icam} 
export ZOOM=${ZOOM:-$zoom}

export QUALITY=90 
export OPTICKS_GEOM=cxr_overview

#[ "$(uname)" == "Darwin" ] && emm=1, || emm=t8,

emm_all=t0        # tilde zero     : (without comma so this is whole number spec)  meaning ~0 (ie 0xffffffff...) for ALL
emm_noglobal=t0,  # tilde 0-th bit : (with comma meaning single bitindex spec) meaning exclude solid 0 (global) 
emm_no8=t8,       # tilde 8-th bit : exclude solid 8 
emm=$emm_all

elv=t
export EMM=${EMM:-$emm}
export ELV=${ELV:-$elv}

## CAUTION: EMM(SBit) and ELV(SBitSet) lingos similar, but not the same. TODO: unify them  

export NAMEPREFIX=cxr_overview_emm_${EMM}_elv_${ELV}_moi_      # MOI gets appended by the executable
export OPTICKS_RELDIR=cam_${ICAM}_tmin_${TMIN}       # this can contain slashes

stamp=$(date +"%Y-%m-%d %H:%M")
version=$(CSGOptiXVersion 2>/dev/null)

export TOPLINE="./cxr_overview.sh    # EYE $EYE MOI $MOI ZOOM $ZOOM stamp $stamp version $version done" 
export BOTLINE=" GEOM $GEOM RELDIR $OPTICKS_RELDIR NAMEPREFIX $NAMEPREFIX SCANNER $SCANNER "

if [ -z "$SCANNER" ]; then 
   vars="stamp version TOPLINE BOTLINE"
   for var in $vars ; do printf "%20s : %s \n" $var "${!var}" ; done
fi

source $DIR/cxr.sh $arg

