#!/bin/bash -l

usage(){ cat << EOU
flight.sh
===============

Whilst first making a flight it is advisable to run with the framelimit envvar 
defined which overrides the framelimit from the --flightconfig option::

    OPTICKS_FLIGHT_FRAMELIMIT=3 EMM=0,1,2,3,4,5,6,7,8,9 PVN=lFasteners_phys flight.sh 


TODO:

1. output directory and jpg naming control
 
2. integrated (at script level) mp4 making using script from env : ffmpeg_mp4_from_jpg.sh 


EOU
}


pvn=${PVN:-lLowerChimney_phys}
emm=${EMM:-~0}                 # SBit::FromString 
size=${SIZE:-2560,1440,1}

bin=OpFlightPathTest

which $bin
pwd

flight="idir=/tmp,prefix=frame,ext=.jpg,scale0=3,scale1=1,framelimit=300"

flightpath-cmd(){ cat << EOC
$bin --targetpvn $pvn --flightconfig $flight -e $emm  $*
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



