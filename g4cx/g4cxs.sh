#!/bin/bash -l 



if [ "$(uname)" == "Linux" ]; then

   gdb G4CXSimulateTest -ex r  

elif [ "$(uname)" == "Darwin" ]; then 

   fold=/tmp/$USER/opticks
   mkdir -p $fold
   cd $fold

   # TODO: scp and run analysis

fi 
