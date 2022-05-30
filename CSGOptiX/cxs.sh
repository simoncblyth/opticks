#!/bin/bash -l 
cxs_msg="=== $BASH_SOURCE : "
case $(uname) in 
   Linux)  argdef="run" ;; 
   Darwin) argdef="ana" ;;
esac 
cxs_arg=${1:-$argdef}

BASH_FOLDER="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

usage(){ cat << EOU
cxs.sh : hybrid rendering/simulation machinery, eg creating 2D ray trace cross sections
========================================================================================

TODO: partition creation and analysis more clearly... currently some 
      stuff comes from metadata written during creation and 
      cannot be updated during analysis

::

    ISEL=0,1,3,4,5 ./cxs.sh ana       # select which boundaries to include in plot 

    XX=-208,208 ZZ=-15.2,15.2 ./cxs.sh 

    NOMASK=1 ./cxs.sh
        Do not mask intersect positions by the limits of the genstep grid
        (so see distant intersects) 

    PVG=1 ./cxs.sh 
        Show the pyvista grid scale 


Two envvars MOI and CEGS configure the gensteps.

The MOI string has form meshName:meshOrdinal:instanceIdx 
and is used to lookup the center extent from the CSGFoundry 
geometry. Examples::

    MOI=Hama
    MOI=Hama:0:0   
    CEGS=16:0:9:200                 # nx:ny:nz:num_photons
    CEGS=16:0:9:200:17700:0:0:200   # nx:ny:nz:num_photons:cx:cy:cz:ew

The CEGS envvar configures an *(nx,ny,nz)* grid from -nx->nx -ny->ny -nz->nz
of integers which are used to mutiply the extent from the MOI center-extent.
The *num_photons* is the number of photons for each of the grid gensteps.

* as the gensteps are currently xz-planar it makes sense to use *ny=0*
* to get a non-distorted jpg the nx:nz should follow the aspect ratio of the frame 

::

    In [1]: sz = np.array( [1920,1080] )
    In [5]: 9*sz/1080
    Out[5]: array([16.,  9.])

Instead of using the center-extent of the MOI selected solid, it is 
possible to directly enter the center-extent in integer mm for 
example adding "17700:0:0:200"

As the extent determines the spacing of the grid of gensteps, it is 
good to set a value of slightly less than the extent of the smallest
piece of geometry to try to get a genstep to land inside. 
Otherwise inner layers can be missed. 

EOU
}


export GEOM=${GEOM:-$geom}

if [ -z "$moi" -o -z "$cegs" -o -z "$ce_offset" -o -z "$ce_scale" -o -z "$gridscale" ]; then 

    echo $cxs_msg the cxs.sh script must now be sourced from other scripts that define a set of local variables
    echo $cxs_msg see for example cxs_solidXJfixture.sh

    [ -z "$moi" ]  && echo $cxs_msg missing moi 
    [ -z "$cegs" ] && echo $cxs_msg missing cegs
    [ -z "$ce_offset" ] && echo $cxs_msg missing ce_offset
    [ -z "$ce_scale" ] && echo $cxs_msg missing ce_scale
    [ -z "$gridscale" ] && echo $cxs_msg missing gridscale 

    exit 1     
fi 

export MOI=${MOI:-$moi}
export CEGS=${CEGS:-$cegs}
export CE_OFFSET=${CE_OFFSET:-$ce_offset}
export CE_SCALE=${CE_SCALE:-$ce_scale}
export GRIDSCALE=${GRIDSCALE:-$gridscale}
export GSPLOT=${GSPLOT:-$gsplot}
export NOTE=${NOTE:-$note}
export NOTE1=${NOTE1:-$note1}

export ISEL=${ISEL:-$isel}
export XX=${XX:-$xx}
export YY=${YY:-$yy}
export ZZ=${ZZ:-$zz}
export OPTICKS_GEOM=$GEOM 


IFS=: read -a cegs_arr <<< "$CEGS"

# quotes on the in variable due to bug fixed in bash 4.3 according to 
# https://stackoverflow.com/questions/918886/how-do-i-split-a-string-on-a-delimiter-in-bash

cegs_elem=${#cegs_arr[@]}

case $cegs_elem in 
   4) echo $cxs_msg 4 element CEGS $CEGS ;; 
   7) echo $cxs_msg 7 element CEGS $CEGS ;; 
   *) echo $cxs_msg ERROR UNEXPECTED $cegs_elem element CEGS $CEGS && exit 1  ;; 
esac


if [ "$(uname)" == "Linux" ]; then
    if [ -n "$cfbase" -a ! -d "$cfbase/CSGFoundry" ]; then

       echo $cxs_msg : ERROR cfbase $cfbase is defined signalling to use a non-standard CSGFoundry geometry 
       echo $cxs_msg : BUT no such CSGFoundry directory exists 
       echo $cxs_msg :
       echo $cxs_msg : Possibilities: 
       echo $cxs_msg :
       echo $cxs_msg : 1. you intended to use the standard geometry but the GEOM $GEOM envvar does not match any of the if branches 
       echo $cxs_msg : 2. you want to use a non-standard geometry but have not yet created it : do so as shown below
       echo $cxs_msg :
       echo $cxs_msg :    \"b7 \; cd ~/opticks/GeoChain\"  
       echo $cxs_msg :    \"gc \; GEOM=$GEOM ./translate.sh\" 
       echo $cxs_msg :   
       exit 1 
    fi 
fi



pkg=CSGOptiX
bin=CSGOptiXSimtraceTest 
export TMPDIR=/tmp/$USER/opticks
export LOGDIR=$TMPDIR/$pkg/$bin
mkdir -p $LOGDIR 


if [ -n "$cfbase" ]; then 
    echo $cxs_msg cfbase $cfbase defined setting CFBASE to override standard geometry default 
    export CFBASE=${CFBASE:-$cfbase}   ## setting CFBASE only appropriate for non-standard geometry 
fi 

export OPTICKS_OUT_FOLD=${CFBASE:-$TMPDIR}/$pkg/$bin/$(SCVDLabel)/$(CSGOptiXVersion)

botline="MOI $MOI CEGS $CEGS GRIDSCALE $GRIDSCALE"


[ -n "$ZOOM" ] && botline="$botline ZOOM $ZOOM"
[ -n "$LOOK" ] && botline="$botline LOOK $LOOK"
[ -n "$XX" ]   && botline="$botline XX $XX"
[ -n "$YY" ]   && botline="$botline YY $YY"
[ -n "$ZZ" ]   && botline="$botline ZZ $ZZ"


topline="cxs.sh MOI $MOI CEGS $CEGS GRIDSCALE $GRIDSCALE"
[ -n "$LOOKCE" ] && topline="$topline LOOKCE $LOOKCE"


export BOTLINE="${BOTLINE:-$botline}"
export TOPLINE="${TOPLINE:-$topline}"


## CAUTION : CURRENTLY THE BOTLINE and TOPLINE from generation which comes from metadata
##  trumps any changes from analysis running
## ... hmm that is kinda not appropriate for cosmetic presentation changes like differnt XX ZZ etc.. 

vars="GEOM CFBASE LOGDIR BASH_FOLDER MOI CE_OFFSET CE_SCALE CXS_CEGS CXS_OVERRIDE_CE GRIDSCALE TOPLINE BOTLINE NOTE GSPLOT ISEL XX YY ZZ FOLD OPTICKS_GEOM OPTICKS_RELDIR OPTICKS_OUT_FOLD"
cxs_dumpvars(){  local var ; local vars=$1 ; shift ; echo $* ; for var in $vars ; do printf "%20s : %s\n" "$var" "${!var}" ; done  ; }
cxs_dumpvars "$vars" initial

cxs_relative_stem()
{
   local path=$1
   local geocache=$HOME/.opticks/geocache/
   local rel 
   case $path in 
      ${geocache}*)  rel=${path/$geocache/} ;;
   esac 
   rel=${rel/\.jpg}
   rel=${rel/\.png}
   echo $rel 
}


cxs_pub()
{
    local msg="$FUNCNAME :"
    local cap_path=$1
    local cap_ext=$2
    local rel_stem=$(cxs_relative_stem ${cap_path})

    if [ "$PUB" == "1" ]; then 
        local extra=""    ## use PUB=1 to debug the paths 
    else
        local extra="_${PUB}" 
    fi 

    local s5p=/env/presentation/${rel_stem}${extra}${cap_ext}
    local pub=$HOME/simoncblyth.bitbucket.io$s5p
    local s5p_line="$s5p 1280px_720px"

    local vars="cap_path cap_ext rel_stem PUB extra s5p pub s5p_line"
    for var in $vars ; do printf "%20s : %s\n" $var "${!var}" ; done  

    mkdir -p $(dirname $pub)

    if [ "$PUB" == "" ]; then 
        echo $msg skipping copy : to do the copy you must set PUB to some descriptive string 
    elif [ "$PUB" == "1" ]; then 
        echo $msg skipping copy : to do the copy you must set PUB to some descriptive string 
    elif [ -f "$pub" ]; then 
        echo $msg published path exists already : NOT COPYING : delete it or set PUB to some different extra string to distinguish the name 
        echo $msg skipping copy : to do the copy you must set PUB to some descriptive string rather than just using PUB=1
    else
        echo $msg copying cap_path to pub 
        cp $cap_path $pub
        echo $msg add s5p_line to s5_background_image.txt
    fi 
}



bin=CSGOptiXSimtraceTest

if [ "$(uname)" == "Linux" ]; then 

    if [ "${cxs_arg}" == "run" ]; then

        cd $LOGDIR 
        $GDB $bin 
        [ $? -ne 0 ] && echo $cxs_msg RUN ERROR at LINENO $LINENO && exit 1 

        source ${bin}_OUTPUT_DIR.sh || exit 1  

    elif [ "${cxs_arg}" == "ana" ]; then 

        cd $LOGDIR 
        source ${bin}_OUTPUT_DIR.sh || exit 1  
        NOGUI=1 ${IPYTHON:-ipython} ${BASH_FOLDER}/tests/$bin.py 

    elif [ "${cxs_arg}" == "run_ana" ]; then 

        cd $LOGDIR 
        $GDB $bin
        [ $? -ne 0 ] && echo $cxs_msg RUN ERROR at LINENO $LINENO && exit 1 
        source ${bin}_OUTPUT_DIR.sh || exit 1  

        if [ -n "$PDB" ]; then
            NOGUI=1 ${IPYTHON:-ipython} --pdb -i ${BASH_FOLDER}/tests/$bin.py 
        else
            NOGUI=1 ${IPYTHON:-ipython}          ${BASH_FOLDER}/tests/$bin.py 
        fi 

    fi

elif [ "$(uname)" == "Darwin" ]; then

    echo $cxs_msg Darwin $(pwd) LINENO $LINENO

    if [ "${cxs_arg}" == "grab" ]; then 
        echo $cxs_msg grab LINENO $LINENO 
        EXECUTABLE=$bin       source cachegrab.sh grab
        EXECUTABLE=CSGFoundry source cachegrab.sh grab
    else
        echo $cxs_msg cxs_arg $cxs_arg LINENO $LINENO
        EXECUTABLE=$bin       source cachegrab.sh env
        cxs_dumpvars "FOLD CFBASE" after cachegrab.sh env

        case ${cxs_arg} in 
           ana) ${IPYTHON:-ipython} --pdb -i ${BASH_FOLDER}/tests/$bin.py  ;; 
           pvcap) source pvcap.sh cap ;;  
           mpcap) source mpcap.sh cap ;;  
           pvpub) source pvcap.sh env ;;
           mppub) source mpcap.sh env ;;
        esac

        echo $cxs_msg cxs_arg $cxs_arg LINENO $LINENO

        if [ "${cxs_arg/pub}" != "${cxs_arg}" ]; then
           cxs_dumpvars "cxs_msg cxs_arg CAP_BASE CAP_REL CAP_PATH CAP_EXT" 
           cxs_pub $CAP_PATH $CAP_EXT 
        else
           echo not pub cxs_arg $cxs_arg
        fi 
    fi  

fi 

exit 0
