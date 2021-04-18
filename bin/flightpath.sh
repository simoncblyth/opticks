#!/bin/bash -l

pvn=${PVN:-lLowerChimney_phys}
eye=${EYE:--1,-1,-1}
emm=${EMM:-0}

size=${SIZE:-2560,1440,1}

bin=OpFlightPathTest

which $bin
pwd

export OPTICKS_FLIGHTPATH_SNAPLIMIT=300    # increase this when sure of targetting and other options

flightpath-cmd(){ cat << EOC
$bin --targetpvn $pvn --eye $eye --enabledmergedmesh $emm --snapoverrideprefix snap-emm-$emm- $*
EOC
}

log=$bin.log
cmd=$(flightpath-cmd $*) 
echo $cmd

printf "\n\n\n$cmd\n\n\n" >> $log 
eval $cmd
rc=$?
printf "\n\n\nRC $rc\n\n\n" >> $log 

echo rc $rc



