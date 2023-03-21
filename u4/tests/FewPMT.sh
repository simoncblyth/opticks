usage(){ cat << EOU
FewPMT.sh
==========

This geomscript is sourced from::

   U4SimtraceTest.sh
   U4SimulateTest.sh

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

version=${VERSION:-0}
pom=${POM:-1}

#layout=two_pmt  
layout=one_pmt
export LAYOUT=$layout

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
export hama_UsePMTOpticalModel=$pom     
export hama_UseNaturalGeometry=$version 

export nnvt_FastCoverMaterial=$fastcover
export nnvt_UsePMTOpticalModel=$pom   
export nnvt_UseNaturalGeometry=$version 

#geomlist=hamaLogicalPMT,nnvtLogicalPMT     # in one_pmt layout get NNVT with this 
#geomlist=nnvtLogicalPMT,hamaLogicalPMT    # in one_pmt layout get HAMA with this
geomlist=nnvtLogicalPMT
#geomlist=hamaLogicalPMT

export FewPMT_GEOMList=$geomlist



vars="BASH_SOURCE VERSION version_desc POM pom_desc GEOM FewPMT_GEOMList LAYOUT "
for var in $vars ; do printf "%-30s : %s \n" "$var" "${!var}" ; done


aspect=1.7777777777777  # 1280/720



case $LAYOUT in 
  one_pmt) loc="upper right" ;; 
        *) loc="skip"        ;; 
esac
export LOC=${LOC:-$loc}      # python ana level presentation, a bit out-of-place ?


if [ "$layout" == "one_pmt" ]; then 

   export U4VolumeMaker_WrapRockWater_Rock_HALFSIDE=210
   export U4VolumeMaker_WrapRockWater_Water_HALFSIDE=200
   export U4VolumeMaker_WrapRockWater_BOXSCALE=$aspect,1,1

elif [ "$layout" == "two_pmt" ]; then 

   export U4VolumeMaker_WrapAroundItem_Rock_HALFSIDE=310  
   export U4VolumeMaker_WrapAroundItem_Water_HALFSIDE=300  
   export U4VolumeMaker_WrapAroundItem_Rock_BOXSCALE=$aspect,1,1
   export U4VolumeMaker_WrapAroundItem_Water_BOXSCALE=$aspect,1,1 

   export ${GEOM}_GEOMWrap=AroundCircle 

   export U4VolumeMaker_MakeTransforms_AroundCircle_radius=250
   export U4VolumeMaker_MakeTransforms_AroundCircle_numInRing=2
   export U4VolumeMaker_MakeTransforms_AroundCircle_fracPhase=0

else
   echo $BASH_SOURCE layout $layout not handled 
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

    export U4Recorder__FAKES_SKIP=1

    # THE BELOW MANUAL CONFIG FOR FAKE DETECTION IS NO LONGER NEEDED
    # FOLLOWING U4Recorder::ClassifyFake ADDITIONS
    # ESPECIALLY FAKE_SURFACE AND FAKE_VV_INNER12 CHECKS.
    #
    # f0=Pyrex/Pyrex:AroundCircle0/hama_body_phys
    # f1=Pyrex/Pyrex:hama_body_phys/AroundCircle0
    # f2=Vacuum/Vacuum:hama_inner1_phys/hama_inner2_phys
    # f3=Vacuum/Vacuum:hama_inner2_phys/hama_inner1_phys
    # f4=Pyrex/Pyrex:AroundCircle1/nnvt_body_phys
    # f5=Pyrex/Pyrex:nnvt_body_phys/AroundCircle1
    # f6=Vacuum/Vacuum:nnvt_inner1_phys/nnvt_inner2_phys
    # f7=Vacuum/Vacuum:nnvt_inner2_phys/nnvt_inner1_phys
    #
    # f8=Pyrex/Pyrex:nnvt_body_phys/nnvt_log_pv 
    # f9=Pyrex/Pyrex:nnvt_log_pv/nnvt_body_phys
    #
    # case $LAYOUT in 
    #    two_pmt) fakes="$f0,$f1,$f2,$f3,$f4,$f5,$f6,$f7" ;;
    #    one_pmt) fakes="$f0,$f1,$f2,$f3,$f4,$f5,$f6,$f7,$f8,$f9" ;;
    # esac
    # export U4Recorder__FAKES="$fakes"
    #
fi 

# standalone access to PMT data 
export PMTSimParamData_BASE=$HOME/.opticks/GEOM/J007/CSGFoundry/SSim/juno


