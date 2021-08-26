#!/bin/bash -l

oe  # opticks env setup

## torchtarget=3153   # DYB
##    following use of GDML auxiliary persisting via geocache
##    can get the default torch target from GDML rather than gaving 
##    to specify on commandline  


bin=G4OKTest 

export CMaterialBridge=INFO
export G4Opticks=INFO
export Opticks=INFO
export OpticksGenstep=INFO
export CGenstepCollector=INFO
export OpticksRun=INFO
 

if [ "$(uname)" == "Linux" ]; then 

gdb -ex r --args $bin $* 

else

lldb_ $bin -o r -- $*

fi 
