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

CIRCLE=0,0,17820,17820 ./xxs.sh 

EOU
}

msg="=== $BASH_SOURCE :"

## X4SolidMaker debug solids

#geom=Orb
#geom=SphereWithPhiSegment 
#geom=SphereWithThetaSegment 
#geom=AdditionAcrylicConstruction
#geom=BoxMinusTubs0
#geom=BoxMinusTubs1
#geom=BoxMinusOrb
#geom=UnionOfHemiEllipsoids
#geom=PolyconeWithMultipleRmin

## PMTSim debug solids 

#geom=polycone
#geom=polycone_zcut-150
#geom=two_tubs_union
#geom=three_tubs_union
#geom=three_tubs_union_zcut-700
#geom=ten_tubs_union_zcut-630
#geom=ten_tubs_union_zcut-420

## PMTSim *maker* solids, always give same solid for each maker

#geom=nnvt_maker_zcut-500.0
#geom=nnvt_maker_zcut-400.0
#geom=nnvt_maker_zcut-350.0
#geom=nnvt_maker_zcut-300.0
#geom=nnvt_maker_zcut-200.0
#geom=nnvt_maker_zcut-183.25

#geom=hama_maker_zcut-500.0
#geom=hama_maker_zcut-400.0
#geom=hama_maker_zcut-350.0
#geom=hama_maker_zcut-300.0
#geom=hama_maker_zcut-183.25

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
#geom=hama_body_solid_prtc_obto
#geom=hama_body_solid_nurs_pdyn
#geom=hama_body_solid_nurs

#geom=hmsk_solidMask
#geom=hmsk_solidMaskTail

#geom=XJfixtureConstruction_YZ
#geom=XJfixtureConstruction_XZ
#geom=XJfixtureConstruction_XY

#geom=XJanchorConstruction_YZ
#geom=XJanchorConstruction_XZ
#geom=XJanchorConstruction_XY

#geom=SJReceiverConstruction_XZ
#geom=AnnulusBoxUnion_YZ
#geom=AnnulusBoxUnion_XY
#geom=AnnulusFourBoxUnion_XY

geom=BoxFourBoxUnion_YX
#geom=BoxThreeBoxUnion_YX

export GEOM=${GEOM:-$geom}
gcn=${GEOM%%_*}   ## name up to the first underscore, assuming use of axis suffix  _XZ _YZ _XY _ZX _ZY _YX 

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

num_pho=10
dx=0
dy=0
dz=0

case $gcn in 
                    Orb)   gridscale=0.15 ;;
   SphereWithPhiSegment)   gridscale=0.15 ;;
   SphereWithThetaSegment) gridscale=0.1  ;;
   BoxMinusOrb)            gridscale=0.12 ;;
   XJfixtureConstruction)  gridscale=0.05 ;;
   XJanchorConstruction)   gridscale=0.05 ;;      
   AnnulusBoxUnion)        gridscale=0.05 ;;     
   SJReceiverConstruction) gridscale=0.05 ;;
   BoxFourBoxUnion)        gridscale=0.07 ;;
   BoxThreeBoxUnion)       gridscale=0.07 ;;
                 *)        gridscale=0.10 ;;
esac

case $GEOM in 
   XJfixtureConstruction_YZ) note="blocky head with ears shape" ;;
   XJfixtureConstruction_XZ) note="appears as three separate rectangles with this slice" ;;
   XJfixtureConstruction_XY) note="pretty celtic cross" ;;
   XJanchorConstruction_YZ)  note="spurious Geant4 intersects on line between cone top and base" ;;
   XJanchorConstruction_XZ)  note="also spurious Geant4 intersects on line between cone top and base, rotational symmetry" ;;
   XJanchorConstruction_XY)  note="circle : z-offset as z-max zero : G4Sphere::DistanceToOut noise" ; dz=-1 ;;
esac

case ${GEOM} in
   ten_tubs_union*) zz=0,-70,-140,-210,-280,-350,-420,-490,-560,-630  ;;
esac

case $gcn in 
   SphereWithPhiSegment)    source SphereWithPhiSegment.sh ;;
   SphereWithThetaSegment)  source SphereWithThetaSegment.sh ;;
   BoxMinusOrb)             source BoxMinusOrb.sh ;;
   XJfixtureConstruction)   source XJfixtureConstruction.sh ;;
   XJanchorConstruction)    source XJanchorConstruction.sh ;;   
esac   
   

case $GEOM in  
   *_XZ) cegs=16:0:9:$dx:$dy:$dz:$num_pho  ;;  
   *_YZ) cegs=0:16:9:$dx:$dy:$dz:$num_pho  ;;  
   *_XY) cegs=16:9:0:$dx:$dy:$dz:$num_pho  ;;  
   *_ZX) cegs=9:0:16:$dx:$dy:$dz:$num_pho  ;;  
   *_ZY) cegs=0:9:16:$dx:$dy:$dz:$num_pho  ;;  
   *_YX) cegs=9:16:0:$dx:$dy:$dz:$num_pho  ;;  
   *_XYZ) cegs=9:16:9:$dx:$dy:$dz:$num_pho ;;  
esac



export GRIDSCALE=${GRIDSCALE:-$gridscale}
export CEGS=${CEGS:-$cegs}
export CXS_RELDIR=${CXS_RELDIR:-$reldir} 
export CXS_OTHER_RELDIR=${CXS_OTHER_RELDIR:-$other_reldir} 
export XX=${XX:-$xx}
export ZZ=${ZZ:-$zz}
export ZZD=${ZZD:-$zzd}
export TOPLINE="x4 ; GEOM=$GEOM ./xxs.sh "
export BOTLINE="$note"
export THIRDLINE="CXS_CEGS=$CXS_CEGS"

check_cegs()
{
    local msg="=== $FUNCNAME :"
    IFS=: read -a cegs_arr <<< "$CEGS"
    local cegs_elem=${#cegs_arr[@]}

    case $cegs_elem in
       4) echo $msg 4 element CEGS $CEGS ;;
       7) echo $msg 7 element CEGS $CEGS ;;
       *) echo $msg ERROR UNEXPECTED $cegs_elem element CEGS $CEGS && return 1  ;;
    esac
    return 0
}
check_cegs || exit 1 




env | grep CXS

arg=${1:-run_ana}

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

dir=$(dirname $BASH_SOURCE)

if [ "${arg/ana}"  != "$arg" ]; then 

    if [ -n "$SCANNER" ]; then 
        ${IPYTHON:-ipython} --pdb $dir/tests/X4IntersectSolidTest.py 
        [ $? -ne 0 ] && echo ana noninteractive error && exit 2
    else
        ${IPYTHON:-ipython} --pdb -i $dir/tests/X4IntersectSolidTest.py 
        [ $? -ne 0 ] && echo ana interactive error && exit 2
    fi
fi 


if [ -n "$PUB" ]; then 

   outdir=$TMP/extg4/X4IntersectSolidTest/$GEOM/X4Intersect/figs 
   reldir=/env/presentation/extg4/X4IntersectSolidTest/$GEOM/X4Intersect/figs 
   pubdir=$HOME/simoncblyth.bitbucket.io$reldir

   pngname=isect_mpplt.png 

   echo $msg outdir $outdir
   echo $msg reldir $reldir
   echo $msg pubdir $pubdir

   if [ ! -d "$pubdir" ]; then 
      mkdir -p $pubdir
   fi 

   cmd="cp $outdir/$pngname $pubdir/$pngname"
   echo $msg cmd $cmd
   eval $cmd
   echo $msh rel $reldir/$pngname
fi 

exit 0 
