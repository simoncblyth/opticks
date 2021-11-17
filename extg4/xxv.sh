#!/bin/bash -l 

usage(){ cat << EOU
xxv.sh : Volume equivalent of xxs.sh 
===============================================

Provides 2D cross section plots of the G4VSolid in a PV tree of solids with structure transforms applied to intersects

EOU
}

msg="=== $BASH_SOURCE :"


# moved to setting in code based on name suffixes  
#
#export JUNO_PMT20INCH_POLYCONE_NECK=ENABLED
#export JUNO_PMT20INCH_SIMPLIFY_CSG=ENABLED
#export JUNO_PMT20INCH_NOT_USE_REAL_SURFACE=ENABLED
#export JUNO_PMT20INCH_PLUS_DYNODE=ENABLED    ## xxs restricted to single solids so this not so useful

#geom=body_phys
#geom=body_phys_nurs
#geom=body_phys_nurs_pcnk
#geom=body_phys_nurs_pcnk_pdyn
#geom=body_phys_nurs_pdyn
#geom=body_phys_pdyn
geom=body_phys_pdyn_pcnk

export GEOM=${GEOM:-$geom}
zcut=${geom#*zcut}
[ "$geom" != "$zcut" ] && zzd=$zcut 
echo geom $geom zcut $geom zzd $zzd

dz=-4
num_pho=10
cegs=9:0:16:0:0:$dz:$num_pho
gridscale=0.10

#zz=190,-162,-195,-210,-275,-350,-365,-420,-450
xx=-254,254

unset CXS_OVERRIDE_CE
export CXS_OVERRIDE_CE=0:0:-130:320   ## fix at the full uncut ce 

reldir=extg4/X4IntersectVolumeTest


export GRIDSCALE=${GRIDSCALE:-$gridscale}
export CXS_CEGS=${CXS_CEGS:-$cegs}
export CXS_RELDIR=${CXS_RELDIR:-$reldir} 
export CXS_OTHER_RELDIR=${CXS_OTHER_RELDIR:-$other_reldir} 
export XX=${XX:-$xx}
export ZZ=${ZZ:-$zz}
export ZZD=${ZZD:-$zzd}

env | grep CXS

arg=${1:-runana}

if [ "${arg/exit}" != "$arg" ]; then
   echo $msg early exit 
   exit 0 
fi

if [ "${arg/run}" != "$arg" ]; then
    $GDB X4IntersectVolumeTest
    [ $? -ne 0 ] && echo run error && exit 1 
fi  
if [ "${arg/dbg}" != "$arg" ]; then
    lldb__ X4IntersectVolumeTest
    [ $? -ne 0 ] && echo run error && exit 1 
fi  
if [ "${arg/ana}"  != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i tests/X4IntersectVolumeTest.py 
    [ $? -ne 0 ] && echo ana error && exit 2
fi 

exit 0 

