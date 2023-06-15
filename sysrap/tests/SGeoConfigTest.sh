#!/bin/bash -l 

elv=Water_solid,Rock_solid
geom=FewPMT

GEOM=${GEOM:-$geom}
ELV=${ELV:-$elv}

GEOM=$GEOM ELV=$ELV  SGeoConfigTest 

GEOM=$GEOM ELV=t$ELV  SGeoConfigTest 

GEOM=$GEOM ELV=~$ELV  SGeoConfigTest 

GEOM=$GEOM ELV=t:$ELV  SGeoConfigTest 

GEOM=$GEOM ELV=\~:$ELV  SGeoConfigTest 

