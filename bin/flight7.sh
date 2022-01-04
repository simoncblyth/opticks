#!/bin/bash -l

msg="=== $BASH_SOURCE :"

usage(){ cat << EOU
flight7.sh
===============

See also flight.sh 


EOU
}

pkg=CSGOptiX
bin=CSGOptiXRenderTest
version=$(CSGOptiXVersion 2>/dev/null)
logdir=/tmp/$USER/opticks/$pkg/$bin/$version

#moi=uni_acrylic1
#moi=Hama:0:1000
#moi=solidXJfixture:55:-3
moi=solidXJfixture:0:-3

period=${PERIOD:-4}
limit=${LIMIT:-600}
scale0=${SCALE0:-3}
scale1=${SCALE1:-0.5}
flight=${FLIGHT:-RoundaboutXY_XZ}
config="flight=$flight,ext=.jpg,scale0=$scale0,scale1=$scale1,framelimit=$limit,period=$period"
nameprefix="${flight}"

export FlightPath=INFO
export FlightPath_scale=${FlightPath_scale:-1}
export MOI=${MOI:-$moi}
export OPTICKS_GEOM=$MOI
export OPTICKS_RELDIR=$flight 


flight-cmd(){ cat << EOC
$bin --flightconfig "$config" --nameprefix "$nameprefix" --flightpathscale ${FlightPath_scale}  $*
EOC
}


flight-init()
{
   which $bin
   pwd

   echo $msg creating output directory logdir: "$logdir"
   mkdir -p "$logdir" 

   cd "$logdir"
}

flight-render-jpg()
{
   local msg="=== $FUNCNAME :"
   local cmd=$(flight-cmd $*) 
   echo $cmd

   local log=$bin.log
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

    local iwd=$PWD

    source ${bin}_OUTPUT_DIR.sh || exit 1
    local evar=${bin}_OUTPUT_DIR ; 
    local outdir=${!evar}
    echo $msg evar $evar outdir $outdir 

    cd "$outdir" 
    pwd

    $jpg2mp4 

    cd $iwd 
    return 0 
}

flight-render()
{
    flight-init
    flight-render-jpg $*
    flight-make-mp4
}


if [ "$(uname)" == "Darwin" ]; then
    cx
    ./cxr_grab.sh mp4 
else
    flight-render $*
fi 

