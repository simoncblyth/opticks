#!/bin/bash -l

usage(){ cat << EOU
flight7.sh
===============

See also flight.sh 


EOU
}

msg="=== $0 :"

export CFBASE=/tmp/$USER/opticks/CSG_GGeo 
[ ! -d "$CFBASE/CSGFoundry" ] && echo $msg ERROR no such directory $CFBASE/CSGFoundry && exit 1


size=${SIZE:-2560,1440,1}  # currently ignored
period=${PERIOD:-4}
limit=${LIMIT:-600}
scale0=${SCALE0:-3}
scale1=${SCALE1:-0.5}
flight=${FLIGHT:-RoundaboutXY}

pkg=CSGOptiX
bin=CSGOptiXFlight
outbase=/tmp/$USER/opticks/$pkg/$bin/$(CSGOptiXVersion)

prefix="${flight}"
outdir="$outbase/$prefix"
config="flight=$flight,ext=.jpg,scale0=$scale0,scale1=$scale1,framelimit=$limit,period=$period"

flight-cmd(){ cat << EOC
$bin --flightconfig "$config" --flightoutdir "$outdir" --nameprefix "$prefix"   $*
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
    flight-render-jpg $*
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
    flight-render $*
fi 



