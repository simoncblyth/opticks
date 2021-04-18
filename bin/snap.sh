#!/bin/bash -l

pvn=${PVN:-lLowerChimney_phys}
eye=${EYE:--1,-1,-1}
emm=${EMM:-0}

snapconfig=${SNAP_CFG:-steps=10,ez0=-1,ez1=5}
size=${SNAP_SIZE:-2560,1440,1}

which OpSnapTest 
pwd




snap-cmd(){ cat << EOC
OpSnapTest --targetpvn $pvn --eye $eye --enabledmergedmesh $emm --snapoverrideprefix snap-emm-$emm- $*
EOC
}

cmd=$(snap-cmd $*) 
echo $cmd

log=OpSnapTest.log
printf "\n\n\n$cmd\n\n\n" >> $log 

eval $cmd
rc=$?
echo rc $rc

printf "\n\n\nRC $rc\n\n\n" >> $log 


 
