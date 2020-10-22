#!/bin/bash -l

oe  # opticks env setup

torchtarget=3153   # DYB

G4Opticks=INFO \
Opticks=INFO \
OpticksGenstep=INFO \
CGenstepCollector=INFO \
OpticksRun=INFO \
  \
lldb_ G4OKTest -o r -- --torchtarget $torchtarget $*

