#!/bin/bash -l 

unset OPTICKS_KEY 
export GGBASE=/tmp/$USER/opticks/ntds3/G4CXOpticks
export GGeo=INFO

export GGeo_deferredCreateGParts_SKIP=1 

GGeoLoadFromDirTest 


