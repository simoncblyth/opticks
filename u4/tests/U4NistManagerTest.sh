#!/bin/bash -l 

bin=U4NistManagerTest

names(){ cat << EON
G4_WATER
G4_AIR
G4_CONCRETE
G4_Pb
EON
}

for name in $(names) ; do
   $bin $name 
done


