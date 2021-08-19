#!/bin/bash -l 
usage(){ cat << EOU

cxr.sh : basis for higher level render scripts
================================================

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

export CFNAME=${CFNAME:-CSG_GGeo}
export CFBASE=/tmp/$USER/opticks/${CFNAME} 
[ ! -d "$CFBASE/CSGFoundry" ] && echo ERROR no such directory $CFBASE/CSGFoundry && exit 1


pkg=CSGOptiX
bin=CSGOptiXRender

# defaults 
cvd=1            # default GPU to use
emm=t8,          # what to include in the GPU geometry 
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
export CAMERATYPE=$CAM    # okc/Camera::Camera default 

vars="CVD EMM MOI EYE TOP SLA CAM TMIN CAMERATYPE"
for var in $vars ; do printf "%10s : %s \n" $var ${!var} ; done 

export BASEDIR=/tmp/$USER/opticks/$pkg/$bin/${CFNAME}/cvd${CVD}/$(CSGOptiXVersion)

# these RELDIR and NAMEPREFIX defaults are typically overridden from higher level script
nameprefix=cxr_${top}_${EMM}_
export NAMEPREFIX=${NAMEPREFIX:-$nameprefix}

reldir=top_${TOP}_
export RELDIR=${RELDIR:-$reldir}

export OUTDIR=${BASEDIR}/${RELDIR}
mkdir -p $OUTDIR

arglist=$OUTDIR/arglist.txt

export LOGDIR=${OUTDIR}.logs
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


make_arglist()
{
    local arglist=$1
    local mname=$CFBASE/CSGFoundry/name.txt  # /tmp/$USER/opticks/CSG_GGeo/CSGFoundry/name.txt  # mesh names
    ls -l $mname
    #cat $mname | grep -v Flange | grep -v _virtual | sort | uniq | perl -ne 'm,(.*0x).*, && print "$1\n" ' -  > $arglist
    cat $mname | grep -v Flange | grep -v _virtual | sort | uniq > $arglist
    ls -l $arglist && cat $arglist 
}


if [ "$MOI" == "ALL" ]; then 
    make_arglist $arglist 
    render --arglist $arglist $*            ## multiple MOI via the arglist 
else
    render $*                               ## single MOI via envvar 

    if [ $? -eq 0 ]; then 
        ls -1rt `find $OUTDIR -name '*.jpg' `
        jpg=$(ls -1rt `find $OUTDIR -name '*.jpg' ` | tail -1)
        echo $msg jpg $jpg 
        ls -l $jpg
        [ -n "$jpg" -a "$(uname)" == "Darwin" ] && open $jpg
    else
        echo $msg non-zero RC from render 
    fi 
fi 

