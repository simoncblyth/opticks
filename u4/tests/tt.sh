#!/bin/bash -l
usage(){ cat << EOU
tt.sh 
======

::

   PLOT=STAMP STAMP_ANNO=1 ~/opticks/u4/tests/tt.sh 
   PLOT=STAMP STAMP_ANNO=1 STAMP_TT=100000,1000 ~/opticks/u4/tests/tt.sh 



EOU
}

DIR=$(dirname $BASH_SOURCE)
N=-1 $DIR/U4SimulateTest.sh tt
