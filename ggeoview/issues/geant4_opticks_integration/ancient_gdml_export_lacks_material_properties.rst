Material Properties
=====================


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




