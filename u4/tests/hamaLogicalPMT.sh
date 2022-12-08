
version=$1

case $version in
  0) echo $BASH_SOURCE FastSim/jPOM ;;
  1) echo $BASH_SOURCE InstrumentedG4OpBoundaryProcess/CustomBoundary ;;
esac

## PMTFastSim/HamamatsuR12860PMTManager declProp config 
export hama_FastCoverMaterial=Cheese  
export hama_UsePMTOpticalModel=1  
export hama_UseNaturalGeometry=$version 

export U4VolumeMaker_WrapAroundItem_Rock_HALFSIDE=750  
export U4VolumeMaker_WrapAroundItem_Water_HALFSIDE=700  
export ${GEOM}_GEOMWrap=AroundCircle 
export U4VolumeMaker_MakeTransforms_AroundCircle_radius=400
export U4VolumeMaker_MakeTransforms_AroundCircle_numInRing=4
export U4VolumeMaker_MakeTransforms_AroundCircle_fracPhase=0

# Simtrace config
export CEGS=16:0:9:10   


