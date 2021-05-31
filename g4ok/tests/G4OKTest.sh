#!/bin/bash -l 

name=CubeCorners
path=$HOME/.opticks/InputPhotons/$name.npy
evar=G4OKTEST_INPUT_PHOTONS_PATH

if [ -f "$path" ]; then 
    export $evar=$path
    echo $evar ${!evar}
fi 

export OpticksGenstep=INFO  # OpticksGenstep::MakeInputPhotonCarrier
export OpticksRun=INFO      # check oac OpticksActionContrl handling of carrier gensteps

bin=$(which G4OKTest)
echo $BASH_SOURCE : bin $bin  

if [ "$(uname)" == "Darwin" ]; then
    lldb__ $bin
else
    gdb $bin
fi

