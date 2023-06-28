#!/bin/bash -l 
usage(){ cat << EOU
SPMT_scan.sh
==============

HUH getting nan again with N_MCT=900 N_SPOL=2 scan.
Cannot reproduce the nan issue, added np.isnan checks
to keep lookout for this. 

::

   PMTIDX=0,2,4 ./SPMT_scan.sh ana 

   # with the default LPMTID_LIST this picks three categories 

EOU
}

REALDIR=$(cd $(dirname $BASH_SOURCE) && pwd )

N_MCT=900 N_SPOL=1  $REALDIR/SPMT_test.sh $*
#N_MCT=90   N_SPOL=1  $REALDIR/SPMT_test.sh $*


