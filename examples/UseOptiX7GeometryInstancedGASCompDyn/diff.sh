#!/bin/bash -l

name=$(basename $PWD)
basis=UseOptiX7GeometryInstancedGASComp

diff $name.cc ../$basis/$basis.cc

srcs=$(ls -1 *.h *.cc)
for src in $srcs ; do 
  if [ -f ../$basis/$src ]; then 
      echo diff ../$basis/$src $src  
      diff ../$basis/$src $src  
  fi 
done



