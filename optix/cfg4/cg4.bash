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

[WORKED AROUND] Issue : old GDML export omits material properties
---------------------------------------------------------------------

Get NULL MPT in loaded model::

    147 void G4GDMLWriteMaterials::MaterialWrite(const G4Material* const materialPtr)
    148 {
    ... 
    163    if (materialPtr->GetMaterialPropertiesTable())
    164    {
    165      PropertyWrite(materialElement, materialPtr);
    166    }

    228 void G4GDMLWriteMaterials::PropertyWrite(xercesc::DOMElement* matElement,
    229                                          const G4Material* const mat)
    230 {
    ...
    241    for (mpos=pmap->begin(); mpos!=pmap->end(); mpos++)
    242    {
    243       propElement = NewElement("property");
    244       propElement->setAttributeNode(NewAttribute("name", mpos->first));
    245       propElement->setAttributeNode(NewAttribute("ref",
    246                                     GenerateName(mpos->first, mpos->second)));


No property elements in the ancient geant4 exported GDML::

    simon:cfg4 blyth$ grep property /usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.gdml

Only that one GDML file amongst the exports, exports were copied over to D:: 

    simon:export blyth$ find . -name '*.gdml'
    ./DayaBay_VGDX_20140414-1300/g4_00.gdml
    simon:export blyth$ pwd
    /usr/local/env/geant4/geometry/export

Anyhow checking geant4.0.2p01/G4GDMLWriteMaterials::MaterialWrite does not write material properties::

    [blyth@ntugrid5 env]$ nuwa-;cd $(nuwa-g4-sdir)


* re-export DYB geometry, checking material properties, old export lacks em  

  * this not so easy, would need to backport recent GDML writer to work with nuwa 
    but the info is in the DAE, and are able to reconstruct G4 materials with 
    the properties for the geocache as done by cfg4- CPropLib, so used this 
    workaround  

  * Actually this work is closely releated to G4DAE exporter and intended 
    eventual revisit to bring up to latest G4 and maybe find way to 
    reduce pain of subsequent such syncing.
    Also note that GDML writer requires special G4 build configuration 
    so if that could be avoided in g4d- ?

  * see also export- 


DONE
-----

* OpticksResource .gdml path handling 
* Break off a CG4 singleton class from cfg4- to hold common G4 components, runmanager etc.. 
* move ggv- tests out of ggeoview- into separate .bash, check the cfg4 tests following refactor 
* add GDML loading 
* workaround lack of MPT in ancient g4 GDML export by converting from the G4DAE export  

TODO
----

* where to slot in CGDMLDetector into the machinery, cli, options, config ?

  * will need step recorder... NumpyEvt etc..
  * maybe bifurcate CCfG4 ? into TEST and FULL  CBaseDetector needed ?  

* try to get contained CG4 to generate smth 
* maybe: split CG4 into separate cg4- package rather than co-locating with cfg4-, cfg4- can depend on cg4-
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
