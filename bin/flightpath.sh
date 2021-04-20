#!/bin/bash -l

pvn=${PVN:-lLowerChimney_phys}
emm=${EMM:-0}
size=${SIZE:-2560,1440,1}

bin=OpFlightPathTest

which $bin
pwd

flight="idir=/tmp,prefix=frame,ext=.jpg,scale0=3,scale1=1,framelimit=300"

export OPTICKS_FLIGHT_FRAMELIMIT=3   ## envvar overrides flightconfig.framelimit 

flightpath-cmd(){ cat << EOC
$bin --targetpvn $pvn --flightconfig $flight --enabledmergedmesh $emm  $*
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



