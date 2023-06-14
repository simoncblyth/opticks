#!/bin/bash -l 

elv=Water_solid,Rock_solid
geom=FewPMT
GEOM=${GEOM:-$geom}
ELV=${ELV:-$elv}

GEOM=$GEOM OPTICKS_ELV_SELECTION=$ELV  SGeoConfigTest 

GEOM=$GEOM OPTICKS_ELV_SELECTION=t$ELV  SGeoConfigTest 


