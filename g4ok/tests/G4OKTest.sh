#!/bin/bash -l 

name=CubeCorners
path=$HOME/.opticks/InputPhotons/$name.npy

if [ -f "$path" ]; then 
    export G4OKTEST_INPUT_PHOTONS_PATH=$path
    echo G4OKTEST_INPUT_PHOTONS_PATH ${G4OKTEST_INPUT_PHOTONS_PATH}
fi 

which G4OKTest 
G4OKTest 

