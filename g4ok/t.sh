#!/bin/bash -l

oe  # opticks env setup

## torchtarget=3153   # DYB
##    following use of GDML auxiliary persisting via geocache
##    can get the default torch target from GDML rather than gaving 
##    to specify on commandline  

G4Opticks=INFO \
Opticks=INFO \
OpticksGenstep=INFO \
CGenstepCollector=INFO \
OpticksRun=INFO \
  \
lldb_ G4OKTest -o r -- $*

