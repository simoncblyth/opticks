# === func-gen- : optix/cfg4/cg4 fgp optix/cfg4/cg4.bash fgn cg4 fgh optix/cfg4
cg4-src(){      echo optix/cfg4/cg4.bash ; }
cg4-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cg4-src)} ; }
cg4-vi(){       vi $(cg4-source) ; }
cg4-env(){      elocal- ; }
cg4-usage(){ cat << EOU

CG4 : Full Opticks/Geant4 integration
======================================

Objective
-----------

Full geometry Geant4 and Opticks integrated, enabling:

* full geometry testing/development/comparison
* Geant4 particle gun control from within interactive Opticks (GGeoView) 

See Also
---------

* cfg4- partial integration for small test geometries only
* export- geometry exporter

Approach
---------

Geant4 and Opticks need to be using the same geometry...
 
* G4DAE for Opticks
* GDML for Geant4 

Standard export- controlled geometry exports include the .gdml
and .dae when they have a "G" and "D" in the path like the 
current standard::

  /usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/

DONE
-----

* OpticksResource .gdml path handling 
* Break off a CG4 singleton class from cfg4- to hold common G4 components, runmanager etc.. 
* move ggv- tests out of ggeoview- into separate .bash, check the cfg4 tests following refactor 

TODO
----

* add GDML loading to CG4 
* split CG4 into separate cg4- package rather than co-locating with cfg4-, cfg4- can depend on cg4-
* bring over, cleanup, simplify G4DAEChroma gdc- (no need for ZMQ) 
  with the customized step collecting Cerenkov and Scintillation processes
* collect other(non-photon producing processes) particle step tree somehow ? 
* step visualization 
* gun control interface, ImGui?  particle palette, shooter mode
* updated JUNO export, both DAE and GDML 


EOU
}
cg4-dir(){ echo $(env-home)/optix/cfg4 ; }
cg4-cd(){  cd $(cg4-dir); }
