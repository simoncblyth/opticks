#!/bin/bash -l

pvn=${PVN:-lLowerChimney_phys}
eye=${EYE:--1,-1,-1}
emm="${EMM:-~8,}"

#config=${SNAP_CFG:-steps=10,ez0=-1,ez1=5}
#size=${SNAP_SIZE:-2560,1440,1}
bin=OpSnapTest 

outbase=$TMP/snap
reldir=${pvn}
nameprefix=${pvn}__${emm}__
outdir=$outbase/$reldir


snap-cmd(){ cat << EOC
$bin --targetpvn $pvn --eye $eye -e "$emm" --snapoutdir $outdir --nameprefix $nameprefix $*
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

snap-grab()
{
    [ -z "$outbase" ] && echo $msg outbase $outbase not defined && return 1 
    local cmd="rsync -rtz --progress P:$outbase/ $outbase/"
    echo $cmd
    eval $cmd
    open $outbase
    return 0 
}

if [ "$(uname)" == "Darwin" ]; then
    snap-grab
else
    snap-render $*
fi 

