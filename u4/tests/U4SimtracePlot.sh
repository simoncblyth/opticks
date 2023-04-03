#!/bin/bash -l 
usage(){ cat << EOU
U4SimtracePlot.sh
===================

Checking U4Navigator Simtrace SEvt 
(which are very different to the old SSimtrace.h solid-centric SEvt) 

These navigator Simtrace SEvt are written by U4SimulateTest.sh 
only when the below envvar is set::

   export U4Recorder__EndOfRunAction_Simtrace=1

EOU
}


geom=FewPMT
evt=999

export VERSION=${N:-0}
export GEOM=${GEOM:-$geom}
export GEOMFOLD=/tmp/$USER/opticks/GEOM/$GEOM
export EVT=${EVT:-$evt}
export FOLD=$GEOMFOLD/U4SimulateTest/ALL/$EVT

DIR=$(dirname $BASH_SOURCE)
${IPYTHON:-ipython} --pdb -i $DIR/U4SimtracePlot.py 



