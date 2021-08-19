#!/bin/bash -l

usage(){ cat << EOU
cxr_flight.sh
===============

See also opticks/bin/flight.sh::

    ./cxr_flight.sh

Before taking flight check a single render with eg::

    ./cxr.sh 
    ./cxr_overview.sh 

Developments here need to follow cxr.sh to some extent. 
This requires the invoked script to set OUTDIR to the 
directory with the jpg renders and to name the jpg smth_00000.jpg 
etc..

EOU
}

msg="=== $0 :"

script=${SCRIPT:-cxr_overview}
period=${PERIOD:-16}
limit=${LIMIT:-1024}
scale0=${SCALE0:-0.6}
scale1=${SCALE1:-0.3}
flight=${FLIGHT:-RoundaboutXY_XZ}

flightconfig="flight=$flight,ext=.jpg,scale0=$scale0,scale1=$scale1,framelimit=$limit,period=$period"

flight-cmd(){ cat << EOC
source ./${script}.sh --flightconfig "$flightconfig" $*
EOC
}

flight-render-jpg()
{
   local msg="=== $FUNCNAME :"
   pwd

   local cmd=$(flight-cmd $*) 
   echo $cmd
   eval $cmd

   local rc=$?
   echo $msg rc $rc
}

flight-make-mp4()
{
    local msg="=== $FUNCNAME :"
    local jpg2mp4=$HOME/env/bin/ffmpeg_jpg_to_mp4.sh
    [ ! -x "$jpg2mp4" ] && echo $msg no jpg2mp4 $jpg2mp4 script && return 1 
    pwd
    $jpg2mp4 
    return 0 
}


flight-render-jpg $*

echo $msg OUTDIR $OUTDIR 
cd $OUTDIR 

flight-make-mp4



