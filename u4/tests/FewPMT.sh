
usage(){ cat << EOU
FewPMT.sh ( formerly hamaLogicalPMT.sh ) 
===========================================

This geomscript is sourced from::

   U4SimtraceTest.sh
   U4SimulateTest.sh


EOU
}


version=${1:-0}
layout=${2:-one_pmt}
pom=1


case $version in
  0) echo $BASH_SOURCE FastSim/jPOM ;;
  1) echo $BASH_SOURCE InstrumentedG4OpBoundaryProcess/CustomBoundary ;;
esac

case $layout in 
  one_pmt) echo layout $layout ;; 
  two_pmt) echo layout $layout ;; 
esac

case $pom in 
   0) echo POM : traditional stop at photocathode PMT has no innards  ;;
   1) echo POM : allow photons into PMT which has innards ;; 
esac

fastcover=Cheese


## PMTFastSim/HamamatsuR12860PMTManager declProp config 

export hama_FastCoverMaterial=$fastcover
export hama_UsePMTOpticalModel=$pom     
export hama_UseNaturalGeometry=$version 

export nnvt_FastCoverMaterial=$fastcover
export nnvt_UsePMTOpticalModel=$pom   
export nnvt_UseNaturalGeometry=$version 

export ${GEOM}_GEOMList=hamaLogicalPMT,nnvtLogicalPMT


if [ "$layout" == "one_pmt" ]; then 

   echo $BASH_SOURCE layout $layout all default U4VolumeMaker settings

elif [ "$layout" == "two_pmt" ]; then 

    export U4VolumeMaker_WrapAroundItem_Rock_HALFSIDE=310  
    export U4VolumeMaker_WrapAroundItem_Water_HALFSIDE=300  
    aspect=1.7777777777777  # 1280/720
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


