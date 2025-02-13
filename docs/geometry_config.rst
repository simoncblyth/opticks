geometry_config
=================

Default location for geometry config script
---------------------------------------------------

The below default location is used my many Opticks scripts to 
configure the geometry by setting the GEOM envvar::

   $HOME/.opticks/GEOM/GEOM.sh 

That location is used by the opticks-t ctests and 
many bash scripts within opticks. 

Clearly when working with multiple geometries it might
prove move convenient for GEOM.sh to source different
scripts depending on the GEOM envvar value.


Example of ~/.opticks/GEOM/GEOM.sh
------------------------------------

::

    #!/bin/bash
    notes(){ cat << EON
    ~/.opticks/GEOM/GEOM.sh
    =========================

    Usage from scripts::

        source $HOME/.opticks/GEOM/GEOM.sh 

    EON
    }

    #geom=V1J009 
    #geom=V1J010 
    geom=V1J011  

    #geom=simpleLArTPC
    #geom=xjfcSolid 
    #geom=Z36
    #geom=RaindropRockAirWater

    export GEOM=$geom
    export ${GEOM}_CFBaseFromGEOM=$HOME/.opticks/GEOM/$GEOM
    export ${GEOM}_GDMLPath=$HOME/.opticks/GEOM/$GEOM/origin.gdml


