#!/bin/bash -l

usage(){ cat << EOU
flight.sh
===============

Creates sequences of raytrace geometry snapshot images in jpg/png/ppm.
Using jpg has the advantage of lossy compression with small file sizes.  


Whilst first making a flight it is advisable to run with the framelimit envvar 
defined which overrides the framelimit from the --flightconfig option::

    OPTICKS_FLIGHT_FRAMELIMIT=3 EMM=~0 PVN=lFasteners_phys flight.sh 

    PVN=lFasteners_phys flight.sh --rtx 1 --cvd 1 


TODO:

1. output directory and jpg naming control
 
2. integrated (at script level) mp4 making using script from env : ffmpeg_mp4_from_jpg.sh 


EOU
}


msg="=== $0 :"
pvn=${PVN:-lLowerChimney_phys}
emm="${EMM:-~0}"                 # SBit::FromString 
size=${SIZE:-2560,1440,1}

bin=OpFlightPathTest

which $bin
pwd

flight="idir=/tmp,prefix=frame,ext=.jpg,scale0=3,scale1=0.5,framelimit=300,period=4"

flight-cmd(){ cat << EOC
$bin --targetpvn $pvn --flightconfig $flight -e "$emm"  $*
EOC
}


log=$bin.log
cmd=$(flight-cmd $*) 
echo $cmd

printf "\n\n\n$cmd\n\n\n" >> $log 
eval $cmd
rc=$?
printf "\n\n\nRC $rc\n\n\n" >> $log 

echo rc $rc



jpg2mp4=$HOME/env/bin/ffmpeg_jpg_to_mp4.sh
[ ! -x "$jpg2mp4" ] && echo $msg no jpg2mp4 $jpg2mp4 script && exit 1

pfx=FlightPath 
cd $TMP/okop/OpFlightPathTest   
pwd

$jpg2mp4 $pfx

