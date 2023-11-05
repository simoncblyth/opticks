geometry_and_input_photon_config
================================

Standard user script for geometry and input photon config
-----------------------------------------------------------

The below standard location is used to config both geometry 
and input photons::

   $HOME/.opticks/GEOM/GEOM.sh 

That location is used by the opticks-t ctests and 
many bash scripts within opticks. 

Clearly when working with multiple geometries it might
prove move convenient for GEOM.sh to source different
scripts depending on the GEOM envvar value.


Example of ~/.opticks/GEOM/GEOM.sh
------------------------------------

::

    #!/bin/bash -l 
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


