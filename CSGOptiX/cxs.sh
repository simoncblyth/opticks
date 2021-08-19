#!/bin/bash -l 
usage(){ cat << EOU

cxs.sh : CSGOptiX simulate 
================================================

EOU
}

msg="=== $BASH_SOURCE :"

export CFNAME=${CFNAME:-CSG_GGeo}
export CFBASE=/tmp/$USER/opticks/${CFNAME} 
[ ! -d "$CFBASE/CSGFoundry" ] && echo ERROR no such directory $CFBASE/CSGFoundry && exit 1


pkg=CSGOptiX
bin=CSGOptiXSimulate

# defaults 
cvd=1            # default GPU to use
emm=t8,          # what to include in the GPU geometry 
top=i0           # hmm difficuly to support other than i0

[ "$(uname)" == "Darwin" ] && cvd=0    # only one GPU on laptop 

export CVD=${CVD:-$cvd}    # --cvd 
export EMM=${EMM:-$emm}    # -e 
export TOP=$top            # evar:TOP? getting TOP=0 from somewhere causing crash

vars="CVD EMM TOP"
for var in $vars ; do printf "%10s : %s \n" $var ${!var} ; done 

export BASEDIR=/tmp/$USER/opticks/$pkg/$bin/${CFNAME}/cvd${CVD}/$(CSGOptiXVersion)

# these RELDIR and NAMEPREFIX defaults are typically overridden from higher level script
nameprefix=cxs_${top}_${EMM}_
export NAMEPREFIX=${NAMEPREFIX:-$nameprefix}

reldir=top_${TOP}_
export RELDIR=${RELDIR:-$reldir}

export OUTDIR=${BASEDIR}/${RELDIR}
mkdir -p $OUTDIR

export LOGDIR=${OUTDIR}.logs
mkdir -p $LOGDIR 
cd $LOGDIR 


DIV=""
[ -n "$GDB" ] && DIV="--" 

simulate-cmd(){ cat << EOC
$GDB $bin $DIV --nameprefix "$NAMEPREFIX" --cvd $CVD -e "$EMM" $* 
EOC
}   

simulate()
{
   local msg="=== $FUNCNAME :"
   which $bin
   pwd

   local log=$bin.log
   local cmd=$(simulate-cmd $*) 
   echo $cmd

   printf "\n\n\n$cmd\n\n\n" >> $log 

   eval $cmd
   local rc=$?

   printf "\n\n\nRC $rc\n\n\n" >> $log 

   echo $msg rc $rc

   return $rc
}

simulate $*     

if [ $? -eq 0 ]; then 
    ls -1rt `find $OUTDIR -name '*.npy' `
    npy=$(ls -1rt `find $OUTDIR -name '*.npy' ` | tail -1)
   echo $msg npy $npy 
   ls -l $npy
else
   echo $msg non-zero RC from simulate
fi 

