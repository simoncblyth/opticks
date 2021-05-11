#!/bin/bash -l

usage(){ cat << EOU

CAUTION : this gets installed by opticks/bin/CMakeLists.txt 
so remember to do so to see updates


Usage::

   snap.sh --rtx 1 --cvd 1 

See also snapscan.sh for varying -e option::

   snapscan.sh --rtx 1 --cvd 1 

EOU
}


pvn=${PVN:-lLowerChimney_phys}
eye=${EYE:--1,-1,-1}
emm="${EMM:-t8,}"

#config=${SNAP_CFG:-steps=10,ez0=-1,ez1=5}
#size=${SNAP_SIZE:-2560,1440,1}
bin=OpSnapTest 

outbase=$TMP/snap
reldir=${pvn}
nameprefix=${pvn}__${emm}__
outdir=$outbase/$reldir


snap-cmd(){ cat << EOC
$GDB $bin --targetpvn $pvn --eye $eye -e $emm --snapoutdir $outdir --nameprefix $nameprefix $*
EOC
}

snap-render()
{
    which $bin 
    pwd
    local cmd=$(snap-cmd $*) 
    echo $cmd
    local log=$bin.log
    printf "\n\n\n$cmd\n\n\n" >> $log 
    eval $cmd
    rc=$?
    echo rc $rc
    printf "\n\n\nRC $rc\n\n\n" >> $log 
}

snap-grab-cmd(){ 
    local tmpdir=$1
    [ ! -d "$tmpdir" ] && echo $msg no tmpdir && return 1 
    local from=P:${tmpdir}/
    local to=${tmpdir}/
    cat << EOC
rsync -rtz --progress $from $to
EOC
#rsync -zarv --progress --include="*/" --include="*.jpg" --include="*.mp4" --exclude="*" "$from" "$to" "
}


snap-grab()
{
    mkdir -p $outbase
    [ $? -ne 0 ] && echo $msg failed to create outbase $outbase && return 1 
    local cmd=$(snap-grab-cmd $outbase)
    echo $cmd
    eval $cmd
    open $outbase
    return 0 
}


if [ "$(uname)" == "Darwin" -o "$1" == "grab" ]; then
    snap-grab
else
    snap-render $*
fi 

