#!/bin/bash  
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

To copy the last jpg render into s5p publication folders rerun the commandline 
with PUB=some_descriptive_string. 
The last render will be opened and a publication path determined and the jpg copied to it. 
To check the path to be used without doing the copy use PUB=1. 
Examples::

   PUB=1                      EYE=1,0,0 ZOOM=1 ./cxr_overview.sh 
   PUB=simple_transform_check EYE=1,0,0 ZOOM=1 ./cxr_overview.sh 


TODO: former support for "--arglist" "--solid_label" "--sizescale" "--size" needs rejig as no longer parse args

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))
SDIR=$PWD

msg="=== $BASH_SOURCE :"

if [ -n "$CFNAME" ]; then
    echo $msg CFNAME $CFNAME is defined : this is used by cxr_geochain.sh and cxr_demo.sh 
    export CFBASE=/tmp/$USER/opticks/${CFNAME}    ## override CFBASE envvar only used when CFNAME defined, eg for demo geometry
    echo $msg CFNAME $CFNAME CFBASE $CFBASE OVERRIDING 
    if [ ! -d "$CFBASE/CSGFoundry" ]; then 
        echo $msg ERROR CFNAME override but no corresponding CSGFoundry directory $CFBASE/CSGFoundry 
        echo $msg TO CREATE NON-STANDARD geometries use \"gc \; GEOM=$(basename $CFNAME) ./run.sh\"  
        return 1
    fi
else
    export ${GEOM}_CFBaseFromGEOM=$HOME/.opticks/GEOM/$GEOM
fi 

defarg="run"
arg=${1:-$defarg}

pkg=CSGOptiX
bin=CSGOptiXRenderTest

# defaults 
emm=t0           # what to include in the GPU geometry : default to t0 ie ~0 which means everything 
moi=sWaterTube   # should be same as lLowerChimney_phys
eye=-1,-1,-1,1   # where to look from, see okc/View::home 
top=i0           # hmm difficuly to support other than i0
sla=             # solid_label selection 
icam=0           # 0:perpective 1:orthographic 2:equirect (2:not supported in CSGOptiX(7) yet)
tmin=0.1         # near in units of extent, so typical range is 0.1-2.0 for visibility, depending on EYE->LOOK distance
zoom=1.0
size=1280,720,1
sizescale=1.5

# [ "$(uname)" == "Darwin" ] && cvd=0    # only one GPU on laptop : BUT are usually just grabbing from remote

export CVDLabel="CVD${CUDA_VISIBLE_DEVICES}" 

export EMM=${EMM:-$emm}    # -e 
export MOI=${MOI:-$moi}    # evar:MOI OR --arglist when MOI=ALL  
export EYE=${EYE:-$eye}    # evar:EYE 
export TOP=$top            # evar:TOP? getting TOP=0 from somewhere causing crash
export SLA="${SLA:-$sla}"  # --solid_label
export ICAM=${ICAM:-$icam} # evar:CAMERATYPE  NB caller script ICAM overrides
export TMIN=${TMIN:-$tmin} # evar:TMIN
export ZOOM=${ZOOM:-$zoom} 
export SIZE=${SIZE:-$size} 
export SIZESCALE=${SIZESCALE:-$sizescale} 



# okc Composition/Camera::Camera needs CAMERATYPE integer 
export CAMERATYPE=$ICAM     
# SGLM needs string sname 
case $ICAM in
  0) export CAM="perspective" ;;
  1) export CAM="orthographic" ;;
esac


export OPTICKS_GEOM=${OPTICKS_GEOM:-$MOI}  # "sweeper" role , used by Opticks::getOutPrefix   

case $(uname) in 
   Darwin) optix_version=70000 ;;   ## HMM: assuming remote using that version
   Linux)  optix_version=$(CSGOptiXVersion 2>/dev/null) ;;
esac


# the OPTICKS_RELDIR and NAMEPREFIX defaults are typically overridden from higher level script
nameprefix=cxr_${top}_${EMM}_${SIZE}_
export NAMEPREFIX=${NAMEPREFIX:-$nameprefix}

reldir=top_${TOP}_
export OPTICKS_RELDIR=${OPTICKS_RELDIR:-$reldir}  
## TODO: the RELDIR is being ignored 

# changed layout
# 1. add GEOM/$GEOM at top 
# 2. remove the CSGOptiX pkg above the bin
#   N[blyth@localhost opticks]$ mv CSGOptiX/CSGOptiXRenderTest GEOM/V1J008/



export PBAS=/tmp/$USER/opticks/  # NB TRAILING SLASH 
export TMPDIR=${PBAS}GEOM/$GEOM
export LOGDIR=$TMPDIR/$bin
mkdir -p $LOGDIR 
cd $LOGDIR 


export OPTICKS_OUT_FOLD=${CFBASE:-$TMPDIR}/$bin/$CVDLabel/${optix_version}
export OPTICKS_OUT_NAME=$MOI
if [ -n "$SCAN" ]; then
    OPTICKS_OUT_NAME=$OPTICKS_OUT_NAME/$SCAN
fi 
export BASE=$OPTICKS_OUT_FOLD/$OPTICKS_OUT_NAME


# Example BASE directory path for scan outputs 
#      /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm
#
# SEventConfig::OutDir 
#    $OPTICKS_OUT_FOLD/$OPTICKS_OUT_NAME  (should be same as BASE)
# 
# SEventConfig::OutPath
#    SEventConfig::OutDir()/stem... 
#



if [ -z "$SCAN" ]; then 

   vars="CUDA_VISIBLE_DEVICES EMM MOI EYE TOP SLA CAM TMIN ZOOM CAMERATYPE OPTICKS_GEOM OPTICKS_RELDIR SIZE SIZESCALE CFBASE OPTICKS_OUT_FOLD OPTICKS_OUT_NAME"
   for var in $vars ; do printf "%10s : %s \n" $var ${!var} ; done 

fi 


if [ -n "$DEBUG" ]; then 
    case $(uname) in 
       Darwin) GDB=lldb__ ;;
       Linux)  GDB=gdb    ;;
    esac
fi 

DIV=""
[ -n "$GDB" ] && DIV="--" 


render()
{
   local msg="=== $FUNCNAME :"

   local log=$bin.log
   local cmd="$GDB $bin" 

   if [ -z "$SCAN" ]; then
      which $bin
      pwd
      echo ==== $BASH_SOURCE $FUNCNAME : $cmd
   fi 

   printf "\n\n\n$cmd\n\n\n" >> $log 

   eval $cmd
   local rc=$?

   printf "\n\n\nRC $rc\n\n\n" >> $log 

   if [ -z "$SCAN" ]; then
       echo ==== $BASH_SOURCE $FUNCNAME : $cmd : rc $rc
   fi
 
   return $rc
}



if [ "$arg" == "run" ]; then

    render 

elif [ "$arg" == "grab" -o "$arg" == "open" -o "$arg" == "grab_open" ]; then 

    source $SDIR/../bin/BASE_grab.sh $arg 

elif [ "$arg" == "jstab" ]; then 

    source $SDIR/../bin/BASE_grab.sh $arg 

elif [ "$arg" == "pub" -o "$arg" == "list" ]; then 

    source $SDIR/../bin/BASE_grab.sh $arg 

elif [ "$arg" == "info" ]; then 

    vars="BASH_SOURCE BASE TMPDIR LOGDIR NAMEPREFIX SDIR optix_version"
    for var in $vars ; do printf " %20s : %s \n" "$var" "${!var}" ; done
fi 

