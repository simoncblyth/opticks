#!/bin/bash -l 
usage(){ cat << EOU
xxs.sh : Geant4 equivalent to OptiX cxs.sh 
===============================================

Provides 2D cross section plots of G4VSolid provided from j/PMTSim. 

The configuration of solid/volume modelling is controlled by 
envvars that are now set by PMTSim::SetEnvironmentSwitches
based on string suffixes to the requested solid or volume names. 
This simplifies bookkeeping during development.  

These name suffix opts should perhaps be moved to a separate opts argument
once developments are nearly finalized. 

+--------+------------------------------------------------+-----------------------------------------------------------------------------------+
| suffix | key                                            | note                                                                              | 
+========+================================================+===================================================================================+
| _pcnk  | JUNO_PMT20INCH_POLYCONE_NECK=ENABLED           | switch now removed, as is now the default                                         | 
| _obto  | JUNO_PMT20INCH_OBSOLETE_TORUS_NECK=ENABLED     | obsolete torus neck, fails without also _prtc                                     |
| _prtc  | JUNO_PMT20INCH_PROFLIGATE_TAIL_CUT=ENABLED     | profligate tail cut                                                               |
| _scsg  | JUNO_PMT20INCH_SIMPLIFY_CSG=ENABLED            |                                                                                   |
| _nurs  | JUNO_PMT20INCH_NOT_USE_REAL_SURFACE=ENABLED    | switch off manager level z-cutting                                                |
| _pdyn  | JUNO_PMT20INCH_PLUS_DYNODE=ENABLED             | adds dynode volumes inside inner2_log so need to look at xxv.sh to see effect     |
+--------+------------------------------------------------+-----------------------------------------------------------------------------------+

EOU
}

msg="=== $BASH_SOURCE :"

## X4GeometryMaker debug solids

#geom=orb
#geom=AdditionAcrylicConstruction
#geom=BoxMinusTubs0
#geom=BoxMinusTubs1
#geom=UnionOfHemiEllipsoids

## PMTSim debug solids 

#geom=polycone
#geom=polycone_zcut-150
#geom=two_tubs_union
#geom=three_tubs_union
#geom=three_tubs_union_zcut-700
#geom=ten_tubs_union_zcut-630
#geom=ten_tubs_union_zcut-420

## PMTSim *maker* solids, always give same solid for each maker

#geom=nnvt_maker_zcut-183.25
#geom=nnvt_maker_zcut-300.0
#geom=nnvt_maker_zcut-350.0
#geom=nnvt_maker_zcut-400.0
#geom=nnvt_maker_zcut-500.0

#geom=hama_maker_zcut-183.25
#geom=hama_maker_zcut-300.0
#geom=hama_maker_zcut-350.0
#geom=hama_maker_zcut-400.0
#geom=hama_maker_zcut-500.0

#geom=pmt_solid
#geom=I
#geom=III
#geom=1_2
#geom=1_3
#geom=_pmt_cut_solid
#geom=pmt_solid
#geom=body_solid
#geom=body_solid_zcut
#geom=body_solid_zcut,body_solid
#geom=inner2_solid_zcut
#geom=pmt_solid_zcut
#geom=body_solid,inner2_solid   


## *manager* solids yield different shapes depending on the string between prefix and options

#geom=nnvt_body_solid
#geom=nnvt_body_solid_nurs

#geom=hama_body_solid
#geom=hama_body_solid_nurs
#geom=hama_body_solid_prtc
geom=hama_body_solid_prtc_obto
#geom=body_solid_nurs_pdyn
#geom=body_solid_nurs


export GEOM=${GEOM:-$geom}
zcut=${GEOM#*zcut}
[ "$GEOM" != "$zcut" ] && zzd=$zcut 
echo geom $geom GEOM $GEOM zcut $zcut zzd $zzd

tmp=/tmp/$USER/opticks
reldir=extg4/X4IntersectTest
fold=$tmp/$reldir 

echo $msg reldir $reldir fold $fold 

other_reldir=CSGOptiX/CSGOptiXSimulateTest
other_fold=$tmp/$other_reldir

if [ -d "$other_fold" ]; then
    other_exists=YES
else
    other_exists=NO
fi  
echo $msg other_reldir $other_reldir other_fold $other_fold other_exists $other_exists

if [ "$GEOM" == "orb" ]; then
    dz=0
    num_pho=10
    cegs=16:0:9:0:0:$dz:$num_pho
    gridscale=0.15

    zz=-100,100
    xx=-100,100
else
    dz=-4
    num_pho=10
    #cegs=16:0:9:0:0:$dz:$num_pho
    #gridscale=0.15
    cegs=9:0:16:0:0:$dz:$num_pho
    gridscale=0.10

    #zz=190,-162,-195,-210,-275,-350,-365,-420,-450
    xx=-254,254

    unset CXS_OVERRIDE_CE
    export CXS_OVERRIDE_CE=0:0:-130:320   ## fix at the full uncut ce 
    # -320-130 = -450  320-130 = 190 
fi 


case ${GEOM} in
   ten_tubs_union*) zz=0,-70,-140,-210,-280,-350,-420,-490,-560,-630  ;;
esac


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
    $GDB X4IntersectSolidTest
    [ $? -ne 0 ] && echo run error && exit 1 
fi  
if [ "${arg/dbg}" != "$arg" ]; then
    lldb__ X4IntersectSolidTest
    [ $? -ne 0 ] && echo run error && exit 1 
fi  
if [ "${arg/ana}"  != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i tests/X4IntersectSolidTest.py 
    [ $? -ne 0 ] && echo ana error && exit 2
fi 

exit 0 
