usage(){ cat << EOU
FewPMT.sh
==========

This geomscript may depending on GEOM be sourced for example from::

   u4/tests/U4SimtraceTest.sh
   u4/tests/U4SimulateTest.sh
   g4cx/tests/G4CXTest.sh


Moved LAYOUT control here to be in common 
between U4SimulateTest.sh and U4SimtraceTest.sh 


What to to do after changing geometry config
----------------------------------------------

1. rerun U4SimtraceTest intersect geometries with:: 

   ./viz.sh runboth 

2. rerun U4SimulateTest simulation results with::

   ./cf.sh    # default run_cf


Which config where ?
----------------------

Any geometry specific config belongs here.
Photon generation while depending on geometry for targetting
is sufficiently independent to make it better handled separately. 

+-------------------+------------------------------+
|  input envvars    |                              |
+===================+==============================+
| VERSION           |  0/1                         |
+-------------------+------------------------------+
| POM               |  0/1 traditional/multifilm   |
+-------------------+------------------------------+

+-------------------+------------------------------+
|  output envvars   |                              |
+===================+==============================+
| LAYOUT            |  one_pmt/two_pmt             |
+-------------------+------------------------------+
| Many envvars      | Config geometry and fakes    |
+-------------------+------------------------------+

EOU
}

version=1
pom=1 
#layout=two_pmt  
layout=one_pmt

VERSION=${VERSION:-$version}
POM=${POM:-$pom}
LAYOUT=${LAYOUT:-$layout}

export LAYOUT

case $VERSION in
  0) version_desc="N=0 unnatural geometry : FastSim/jPOM" ;;
  1) version_desc="N=1 natural geometry : CustomBoundary" ;;
esac

case $POM in 
  0) pom_desc="POM:$POM traditional stop at photocathode : PMT with no innards"  ;;
  1) pom_desc="POM:$POM allow photons into PMT which has innards" ;; 
esac

fastcover=Cheese

## PMTSim declProp config of the PMTManager

export hama_FastCoverMaterial=$fastcover
export nnvt_FastCoverMaterial=$fastcover

export hama_UsePMTOpticalModel=$pom     
export nnvt_UsePMTOpticalModel=$pom   

export hama_UsePMTNaturalGeometry=$version 
export nnvt_UsePMTNaturalGeometry=$version 

#geomlist=hamaLogicalPMT,nnvtLogicalPMT     # in one_pmt layout get NNVT with this 
#geomlist=nnvtLogicalPMT,hamaLogicalPMT    # in one_pmt layout get HAMA with this
#geomlist=nnvtLogicalPMT
#geomlist=hamaLogicalPMT
#geomlist=tub3LogicalPMT       # A/B match with circle_inwards_100 

#geomlist=hmskLogicMaskVirtual
#geomlist=nmskLogicMaskVirtual  
#geomlist=xjacLogical
geomlist=xjfcLogical

export FewPMT_GEOMList=$geomlist

# OBSERVATIONS ALL WITH SEventConfig::PropagateEpsilon default of 0.05 
#delta=1e-3   # YUCK WITH tub3 G4CXTest.sh : DEGENERATE DEFAULT IN ORIGINAL C++ 
#delta=1e-2   # ALSO YUCK WITH tub3 G4CXTest.sh
#delta=4e-2   # TODO: TRY THIS : IS IT REALLY JUST WHEN DIP BELOW PropagateEpsilon THAT THINGS FALL TO PIECES ?
#delta=5e-2   # OK WITH tub3 G4CXTest.sh  THIS EQUALS SEventConfig::PropagateEpsilon 
#delta=1e-1   # OK WITH tub3 G4CXTest.sh
delta=1       # OK WITH tub3 G4CXTest.sh   
export Tub3inchPMTV3Manager__VIRTUAL_DELTA_MM=$delta


#magic=0.01    # decrease to try to get LPMT apex degeneracy issue to appear standalone 
#magic=0.04     # just less than PropagateEpsilon
#magic=0.05    # initial default in original C++ of both HamamatsuMaskManager and NNVTMaskManager
magic=0.1      # TRY A CONSERVATIVE DOUBLING OF THE CLEARANCE 
#magic=1       # CHECK ITS WORKING BY MAKING EASILY VISIBLE IN simtrace plot : yes, but this could cause overlaps 
export HamamatsuMaskManager__MAGIC_virtual_thickness_MM=$magic
export NNVTMaskManager__MAGIC_virtual_thickness_MM=$magic

#export U4Tree__DISABLE_OSUR_IMPLICIT=1  # HMM: THIS IS SOMEWHAT OF A HIDDEN PLACE TO DO THIS ? 


vars="BASH_SOURCE VERSION version_desc POM pom_desc GEOM FewPMT_GEOMList LAYOUT "
for var in $vars ; do printf "%-30s : %s \n" "$var" "${!var}" ; done


aspect=1.7777777777777  # 1280/720



case $LAYOUT in 
  one_pmt) loc="upper right" ;; 
        *) loc="skip"        ;; 
esac
export LOC=${LOC:-$loc}      # python ana level presentation, a bit out-of-place ?


if [ "$LAYOUT" == "one_pmt" ]; then 

   export U4VolumeMaker_WrapRockWater_Rock_HALFSIDE=220     # formerly 210
   export U4VolumeMaker_WrapRockWater_Water_HALFSIDE=210    # formerly 200
   export U4VolumeMaker_WrapRockWater_BOXSCALE=$aspect,$aspect,1

elif [ "$LAYOUT" == "two_pmt" ]; then 

   export U4VolumeMaker_WrapAroundItem_Rock_HALFSIDE=310  
   export U4VolumeMaker_WrapAroundItem_Water_HALFSIDE=300  
   export U4VolumeMaker_WrapAroundItem_Rock_BOXSCALE=$aspect,1,1
   export U4VolumeMaker_WrapAroundItem_Water_BOXSCALE=$aspect,1,1 

   export ${GEOM}_GEOMWrap=AroundCircle 

   export U4VolumeMaker_MakeTransforms_AroundCircle_radius=250
   export U4VolumeMaker_MakeTransforms_AroundCircle_numInRing=2
   export U4VolumeMaker_MakeTransforms_AroundCircle_fracPhase=0

else
   echo $BASH_SOURCE LAYOUT $LAYOUT not handled 
fi 

# Simtrace config
export CEGS=16:0:9:10   


if [ "$VERSION" == "0" ]; then 

    # jPOM config
    ModelTriggerSimple=0  # default 
    ModelTriggerBuggy=1
    ModelTrigger_IMPL=$ModelTriggerSimple
    #ModelTrigger_IMPL=$ModelTriggerBuggy

    export junoPMTOpticalModel__PIDX_ENABLED=1
    export junoPMTOpticalModel__ModelTrigger_IMPL=$ModelTrigger_IMPL
    export G4FastSimulationManagerProcess_ENABLE=1  

    #export U4Recorder__FAKES_SKIP=1
    #export U4Recorder__ClassifyFake_FindPV_r=1  ## this is slow, but it finds fakes better, use in standalone testing 
    ## export U4Recorder__FAKES="$fakes"  formerly used manual config of fakes skipping

fi 

# standalone access to PMT data 
#export PMTSimParamData_BASE=$HOME/.opticks/GEOM/J007/CSGFoundry/SSim/juno
export PMTSimParamData_BASE=$HOME/.opticks/GEOM/V1J009/CSGFoundry/SSim/extra/jpmt



