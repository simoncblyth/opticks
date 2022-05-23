#!/bin/bash -l 

eye=-0.4,0,0
moi=-1

export EYE=${EYE:-$eye} 
export MOI=${MOI:-$moi}

name=cx$MOI.jpg


if [ "$(uname)" == "Linux" ]; then
   G4CXRenderTest 
elif [ "$(uname)" == "Darwin" ]; then 
   fold=/tmp/$USER/opticks

   mkdir -p $fold
   cd $fold
   scp P:$fold/$name .

   open $name  

fi 
