#!/bin/bash -l 

case $(uname) in 
   Linux)  argdef="run" ;; 
   Darwin) argdef="ana" ;;
esac 
arg=${1:-$argdef}

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

msg="=== $BASH_SOURCE : "

export GEOM=${GEOM:-$geom}

if [ -z "$moi" -o -z "$cegs" -o -z "$ce_offset" -o -z "$ce_scale" -o -z "$gridscale" ]; then 

    echo $msg the cxs.sh script must now be sourced from other scripts that define a set of local variables
    echo $msg see for example cxs_solidXJfixture.sh

    [ -z "$moi" ]  && echo $msg missing moi 
    [ -z "$cegs" ] && echo $msg missing cegs
    [ -z "$ce_offset" ] && echo $msg missing ce_offset
    [ -z "$ce_scale" ] && echo $msg missing ce_scale
    [ -z "$gridscale" ] && echo $msg missing gridscale 

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
   4) echo $msg 4 element CEGS $CEGS ;; 
   7) echo $msg 7 element CEGS $CEGS ;; 
   *) echo $msg ERROR UNEXPECTED $cegs_elem element CEGS $CEGS && exit 1  ;; 
esac







if [ "$(uname)" == "Linux" ]; then
    if [ -n "$cfbase" -a ! -d "$cfbase/CSGFoundry" ]; then

       echo $msg : ERROR cfbase $cfbase is defined signalling to use a non-standard CSGFoundry geometry 
       echo $msg : BUT no such CSGFoundry directory exists 
       echo $msg :
       echo $msg : Possibilities: 
       echo $msg :
       echo $msg : 1. you intended to use the standard geometry but the GEOM $GEOM envvar does not match any of the if branches 
       echo $msg : 2. you want to use a non-standard geometry but have not yet created it : do so as shown below
       echo $msg :
       echo $msg :    \"b7 \; cd ~/opticks/GeoChain\"  
       echo $msg :    \"gc \; GEOM=$GEOM ./translate.sh\" 
       echo $msg :   
       exit 1 
    fi 
fi



pkg=CSGOptiX
bin=CSGOptiXSimtraceTest 
export TMPDIR=/tmp/$USER/opticks
export LOGDIR=$TMPDIR/$pkg/$bin
mkdir -p $LOGDIR 


if [ -n "$cfbase" ]; then 
    echo $msg cfbase $cfbase defined setting CFBASE to override standard geometry default 
    export CFBASE=${CFBASE:-$cfbase}   ## setting CFBASE only appropriate for non-standard geometry 
fi 

export OPTICKS_OUT_FOLD=${CFBASE:-$TMPDIR}/$pkg/$bin/$(SCVDLabel)/$(CSGOptiXVersion)

botline="MOI $MOI CEGS $CXS_CEGS GRIDSCALE $GRIDSCALE"


[ -n "$ZOOM" ] && botline="$botline ZOOM $ZOOM"
[ -n "$LOOK" ] && botline="$botline LOOK $LOOK"
[ -n "$XX" ]   && botline="$botline XX $XX"
[ -n "$YY" ]   && botline="$botline YY $YY"
[ -n "$ZZ" ]   && botline="$botline ZZ $ZZ"

topline="cxs.sh MOI $MOI CXS_CEGS $CXS_CEGS GRIDSCALE $GRIDSCALE"

export BOTLINE="${BOTLINE:-$botline}"
export TOPLINE="${TOPLINE:-$topline}"


## CAUTION : CURRENTLY THE BOTLINE and TOPLINE from generation which comes from metadata
##  trumps any changes from analysis running
## ... hmm that is kinda not appropriate for cosmetic presentation changes like differnt XX ZZ etc.. 

vars="GEOM CFBASE LOGDIR BASH_FOLDER MOI CE_OFFSET CE_SCALE CXS_CEGS CXS_OVERRIDE_CE GRIDSCALE TOPLINE BOTLINE NOTE GSPLOT ISEL XX YY ZZ FOLD OPTICKS_GEOM OPTICKS_RELDIR OPTICKS_OUT_FOLD"
dumpvars(){  local var ; local vars=$1 ; shift ; echo $* ; for var in $vars ; do printf "%20s : %s\n" $var ${!var} ; done  ; }
dumpvars "$vars" initial


bin=CSGOptiXSimtraceTest

if [ "$(uname)" == "Linux" ]; then 

    if [ "$arg" == "run" ]; then

        cd $LOGDIR 
        $GDB $bin 
        [ $? -ne 0 ] && echo $msg RUN ERROR at LINENO $LINENO && exit 1 

        source ${bin}_OUTPUT_DIR.sh || exit 1  

    elif [ "$arg" == "ana" ]; then 

        cd $LOGDIR 
        source ${bin}_OUTPUT_DIR.sh || exit 1  
        NOGUI=1 ${IPYTHON:-ipython} ${BASH_FOLDER}/tests/$bin.py 

    elif [ "$arg" == "run_ana" ]; then 

        cd $LOGDIR 
        $GDB $bin
        [ $? -ne 0 ] && echo $msg RUN ERROR at LINENO $LINENO && exit 1 
        source ${bin}_OUTPUT_DIR.sh || exit 1  

        if [ -n "$PDB" ]; then
            NOGUI=1 ${IPYTHON:-ipython} --pdb -i ${BASH_FOLDER}/tests/$bin.py 
        else
            NOGUI=1 ${IPYTHON:-ipython}          ${BASH_FOLDER}/tests/$bin.py 
        fi 

    fi

elif [ "$(uname)" == "Darwin" ]; then

    echo $msg Darwin $(pwd)

    if [ "${arg/grab}" != "$arg" ]; then 
        echo $msg grab  
        EXECUTABLE=$bin       source cachegrab.sh grab
        EXECUTABLE=CSGFoundry source cachegrab.sh grab
    fi  

    if [ "${arg/ana}" != "$arg" ]; then 
        echo $msg ana
        EXECUTABLE=$bin       source cachegrab.sh env
        dumpvars "FOLD CFBASE" after cachegrab.sh env

        ${IPYTHON:-ipython} --pdb -i ${BASH_FOLDER}/tests/$bin.py 
    fi 

fi 

exit 0
