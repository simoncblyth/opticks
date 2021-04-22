#!/bin/bash -l

usage(){ cat << EOU
flight.sh
===============

See also 

* docs/misc/making_flightpath_raytrace_movies.rst


Creates sequences of raytrace geometry snapshot images in jpg/png/ppm.
Using jpg has the advantage of lossy compression with small file sizes.  


Whilst first making a flight it is advisable to run with the framelimit envvar 
defined which overrides the framelimit from the --flightconfig option::

    OPTICKS_FLIGHT_FRAMELIMIT=3 EMM=~0 PVN=lFasteners_phys flight.sh 

    PVN=lFasteners_phys flight.sh --rtx 1 --cvd 1 

    PVN=lFasteners_phys flight.sh --rtx 1 --cvd 1 

    PERIOD=8 EMM=~5, PVN=lFasteners_phys flight.sh --rtx 1 --cvd 1 

    PERIOD=8 PVN=lLowerChimney_phys flight.sh --rtx 1 --cvd 1 


TODO:

0. use a nameprefix reldir to keep all the files from a flight.sh organized
1. named eye-look-up flightpath input arrays selected by config from .opticks/flightpath input dir 

   * use this for some XZ plane rotation

2. things missing from metadata json output 

   * geocache digest, 
   * GPU name, cvd, RTX setting  etc..


EOU
}

msg="=== $0 :"
pvn=${PVN:-lLowerChimney_phys}
emm="${EMM:-~0}"                 # SBit::FromString 
size=${SIZE:-2560,1440,1}
period=${PERIOD:-4}
limit=${LIMIT:-300}
scale0=${SCALE0:-3}
scale1=${SCALE1:-0.5}
name=${NAME:RoundaboutXY}

outbase="$TMP/flight"
bin=OpFlightPathTest

prefix="${name}__${pvn}__${emm}__${period}__"
outdir="$outbase/$prefix"
config="name=$name,ext=.jpg,scale0=$scale0,scale1=$scale1,framelimit=$limit,period=$period"

flight-cmd(){ cat << EOC
$bin --targetpvn $pvn --flightconfig "$config" --flightoutdir "$outdir" --nameprefix "$prefix" -e "$emm"  $*
EOC
}

flight-render-jpg()
{
   local msg="=== $FUNCNAME :"
   which $bin
   pwd

   echo $msg creating output directory outdir: "$outdir"
   mkdir -p "$outdir" 

   local log=$bin.log
   local cmd=$(flight-cmd $*) 
   echo $cmd

   printf "\n\n\n$cmd\n\n\n" >> $log 
   eval $cmd
   local rc=$?
   printf "\n\n\nRC $rc\n\n\n" >> $log 

   echo $msg rc $rc
}

flight-make-mp4()
{
    local msg="=== $FUNCNAME :"
    local jpg2mp4=$HOME/env/bin/ffmpeg_jpg_to_mp4.sh
    [ ! -x "$jpg2mp4" ] && echo $msg no jpg2mp4 $jpg2mp4 script && return 1 

    cd "$outdir" 
    pwd
    $jpg2mp4 "$prefix"

    return 0 
}

flight-render()
{
    flight-render-jpg
    flight-make-mp4
}

flight-grab()
{
    [ -z "$outbase" ] && echo $msg outbase $outbase not defined && return 1 

    local cmd="rsync -rtz --progress P:$outbase/ $outbase/"
    echo $cmd
    eval $cmd
    open $outbase
    return 0 
}


if [ "$(uname)" == "Darwin" ]; then
    flight-grab
else
    flight-render
fi 



