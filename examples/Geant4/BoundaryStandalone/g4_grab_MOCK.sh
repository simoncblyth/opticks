#!/bin/bash -l 

msg="=== $BASH_SOURCE :"

g4-
srcdir=$(g4-prefix).build/$(g4-name)/source

if [ -f "G4OpBoundaryProcess_MOCK.hh" -o -f "G4OpBoundaryProcess_MOCK.cc"  ]; then 
   echo $msg already grabbed  G4OpBoundaryProcess into G4OpBoundaryProcess_MOCK 
   exit 1  
fi 

cp $srcdir/processes/optical/include/G4OpBoundaryProcess.hh G4OpBoundaryProcess_MOCK.hh
cp $srcdir/processes/optical/src/G4OpBoundaryProcess.cc     G4OpBoundaryProcess_MOCK.cc

srcs="G4OpBoundaryProcess_MOCK.hh G4OpBoundaryProcess_MOCK.cc"

for src in $srcs 
do 
    perl -pi -e 's,G4OpBoundaryProcess,G4OpBoundaryProcess_MOCK,g' $src
done


exit 0
