G4/Op Integration Overview
============================

Objectives
-----------

Full geometry Geant4 and Opticks integrated, enabling:

* full geometry testing/development/comparison
* Geant4 particle gun control from within interactive Opticks (GGeoView) 
* operational without CUDA capable GPU, using Geant4 simulation and OpenGL viz
* drastically faster operation when have  CUDA capable GPU 

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
* add GDML loading 
* workaround lack of MPT in ancient g4 GDML export by converting from the G4DAE export  
* collect other(non-photon producing processes) particle step tree into nopstep buffers

* split G4 geometry handling into TEST and FULL using a CDetector based specialized with:

  * CTestDetector for simple partial geometries
  * CGDMLDetector for full GDML loaded geometries 

DEBUGGING
----------

* nopstep visualization 

TODO
----

* pmt test broken by g4gun generalizations

* CPU indexing, to support non-CUDA capable nodes 

  * also currently the indexing of G4 saved events gets done at every viz

* where to slot in CG4/CGDMLDetector into the machinery, cli, options, config ?

* bring over, cleanup, simplify G4DAEChroma gdc- (no need for ZMQ) 
  with the customized step collecting Cerenkov and Scintillation processes

* gun control interface, ImGui?  particle palette, shooter mode
* updated JUNO export, both DAE and GDML 



