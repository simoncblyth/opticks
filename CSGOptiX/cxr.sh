#!/bin/bash -l 
usage(){ cat << EOU

cxr.sh : basis for higher level render scripts using CSGOptiXRenderTest
=========================================================================

This is typically invoked via higher level scripts, see::

   ./cxr_demo.sh 
   ./cxr_solid.sh 
   ./cxr_view.sh 
   ./cxr_overview.sh 

TODO: revive MOI=ALL controlled from a higher level script, 
MOI=ALL creates an arglist file and uses the --arglist option 
to create a sequence of renders at the positions specified in 
the arglist all from a single load of the geometry.  
So a single run creates multiple different snaps of different
parts of a geometry.


EOU
}

msg="=== $BASH_SOURCE :"

if [ -n "$CFNAME" ]; then
    export CFBASE=/tmp/$USER/opticks/${CFNAME}    ## override CFBASE envvar only used when CFNAME defined, eg for demo geometry
    echo $msg CFNAME $CFNAME CFBASE $CFBASE OVERRIDING 
    if [ ! -d "$CFBASE/CSGFoundry" ]; then 
        echo $msg ERROR CFNAME override but no corresponding CSGFoundry directory $CFBASE/CSGFoundry 
        echo $msg TO CREATE NON-STANDARD geometries use \"gc \; GEOM=$(basename $CFNAME) ./run.sh\"  
        exit 1
    fi
else
    unset CFBASE
    CFNAME=CSG_GGeo
fi 

pkg=CSGOptiX
bin=CSGOptiXRenderTest

# defaults 
cvd=1            # default GPU to use
emm=t0           # what to include in the GPU geometry : default to t0 ie ~0 which means everything 
moi=sWaterTube   # should be same as lLowerChimney_phys
eye=-1,-1,-1,1   # where to look from, see okc/View::home 
top=i0           # hmm difficuly to support other than i0
sla=             # solid_label selection 
cam=0            # 0:perpective 1:orthographic 2:equirect (2:not supported in CSGOptiX(7) yet)
tmin=0.1         # near in units of extent, so typical range is 0.1-2.0 for visibility, depending on EYE->LOOK distance
zoom=1.0

[ "$(uname)" == "Darwin" ] && cvd=0    # only one GPU on laptop 

export CVD=${CVD:-$cvd}    # --cvd 
export EMM=${EMM:-$emm}    # -e 
export MOI=${MOI:-$moi}    # evar:MOI OR --arglist when MOI=ALL  
export EYE=${EYE:-$eye}    # evar:EYE 
export TOP=$top            # evar:TOP? getting TOP=0 from somewhere causing crash
export SLA="${SLA:-$sla}"  # --solid_label
export CAM=${CAM:-$cam}    # evar:CAMERATYPE
export TMIN=${TMIN:-$tmin} # evar:TMIN
export ZOOM=${ZOOM:-$zoom} 
export CAMERATYPE=$CAM     # okc/Camera::Camera default 
export OPTICKS_GEOM=${OPTICKS_GEOM:-$MOI}  # "sweeper" role , used by Opticks::getOutPrefix   

vars="CVD EMM MOI EYE TOP SLA CAM TMIN ZOOM CAMERATYPE OPTICKS_GEOM OPTICKS_RELDIR"
for var in $vars ; do printf "%10s : %s \n" $var ${!var} ; done 

optix_version=$(CSGOptiXVersion 2>/dev/null)


# the OPTICKS_RELDIR and NAMEPREFIX defaults are typically overridden from higher level script
nameprefix=cxr_${top}_${EMM}_
export NAMEPREFIX=${NAMEPREFIX:-$nameprefix}

reldir=top_${TOP}_
export OPTICKS_RELDIR=${OPTICKS_RELDIR:-$reldir}  


export LOGDIR=/tmp/$USER/opticks/$pkg/$bin
mkdir -p $LOGDIR 
cd $LOGDIR 



DIV=""
[ -n "$GDB" ] && DIV="--" 

render-cmd(){ cat << EOC
$GDB $bin $DIV --nameprefix "$NAMEPREFIX" --cvd $CVD -e "$EMM" --solid_label "$SLA" $* 
EOC
}   

render()
{
   local msg="=== $FUNCNAME :"
   which $bin
   pwd

   local log=$bin.log
   local cmd=$(render-cmd $*) 
   echo $cmd

   printf "\n\n\n$cmd\n\n\n" >> $log 

   eval $cmd
   local rc=$?

   printf "\n\n\nRC $rc\n\n\n" >> $log 

   echo $msg rc $rc

   return $rc
}

if [ -n "$ARGLIST" ] ; then 

    echo $msg MOI $MOI ARGLIST $ARGLIST
    render --arglist $ARGLIST $*            ## effectively multiple MOI via the arglist 
    rc=$?

else
    render $*                               ## single MOI via envvar 
    rc=$?
fi


if [ $rc -eq 0 ]; then 

    source CSGOptiXRenderTest_OUTPUT_DIR.sh || exit 1  
    outdir=$CSGOptiXRenderTest_OUTPUT_DIR 

    if [ -n "$outdir" ]; then 
        ls -1rt `find $outdir -name '*.jpg' `
        jpg=$(ls -1rt `find $outdir -name '*.jpg' ` | tail -1)
        echo $msg jpg $jpg 
        ls -l $jpg


        [ -n "$jpg" -a "$(uname)" == "Darwin" ] && open $jpg

        if [ -n "$jpg" -a "$(uname)" == "Darwin" -a -n "$PUB" ]; then 

            if [ "$PUB" == "1" ]; then 
                ext=""
            else
                ext="_${PUB}" 
            fi 

            rel=${jpg/\/tmp\/$USER\/opticks\//}  
            rel=${rel/\.jpg}

            s5p=/env/presentation/${rel}${ext}.jpg
            pub=$HOME/simoncblyth.bitbucket.io$s5p

            echo $msg jpg $jpg
            echo $msg rel $rel
            echo $msg ext $ext
            echo $msg pub $pub
            echo $msg s5p $s5p 1280px_720px 
            mkdir -p $(dirname $pub)

            if [ -f "$pub" ]; then 
                echo $msg published path exists already : NOT COPYING : set PUB to an ext string to distinguish the name or more permanently arrange for a different path   
            else
                echo $msg copying jpg to pub 
                cp $jpg $pub
                echo $msg add s5p to s5_background_image.txt
            fi 
        fi 

    else
        echo $msg outdir not defined 
    fi 
else
    echo $msg non-zero RC from render 
fi 



