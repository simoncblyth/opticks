GDML_matrix_values_truncation
===============================

Truncation of matrix values appears to have been fixed as of 10.3, 2 years ago::

    G4GDML: Reading '/usr/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/960713d973bd4be73b1b7d9aa4838c3e/1/g4ok.gdml'...
    G4GDML: Reading definitions...

    -------- EEEE ------- G4Exception-START -------- EEEE -------
    *** G4Exception : InvalidSize
          issued by : G4GDMLEvaluator::DefineMatrix()
    Matrix 'RINDEX0x10e292390' is not filled correctly!
    *** Fatal Exception *** core dump ***
    -------- EEEE -------- G4Exception-END --------- EEEE -------

::

     04   <define>
      5     <matrix coldim="2" name="EFFICIENCY0x10e2939d0" values="2.034e-06 0.5 4.136e-06 0.5"/>
      6     <matrix coldim="2" name="RINDEX0x10e2933c0" values="2.034e-06 1.49 4.136e-06 1.49"/>
      7     <matrix coldim="2" name="RINDEX0x10e292390" values="2.034e-06 1.3435 2.068e-06 1.344 2.103e-06 1.3445 2.139e-06 1.345 2.177e-06 1.3455 2.216e-06 1.346 2"/>
      8     <matrix coldim="2" name="RINDEX0x10e2906c0" values="2.034e-06 1 2.068e-06 1 2.103e-06 1 2.139e-06 1 2.177e-06 1 2.216e-06 1 2.256e-06 1 2.298e-06 1 2.34"/>
      9   </define>
     10 
     // using :set ruler see that matrix values are truncated at 100 chars 


/usr/local/opticks/externals/g4/geant4_10_02_p01/source/persistency/gdml/src/G4GDMLWriteMaterials.cc::

    210 void G4GDMLWriteMaterials::PropertyVectorWrite(const G4String& key,
    211                            const G4PhysicsOrderedFreeVector* const pvec)
    212 {
    213    const G4String matrixref = GenerateName(key, pvec);
    214    xercesc::DOMElement* matrixElement = NewElement("matrix");
    215    matrixElement->setAttributeNode(NewAttribute("name", matrixref));
    216    matrixElement->setAttributeNode(NewAttribute("coldim", "2"));
    217    std::ostringstream pvalues;
    218    for (size_t i=0; i<pvec->GetVectorLength(); i++)
    219    {
    220        if (i!=0)  { pvalues << " "; }
    221        pvalues << pvec->Energy(i) << " " << (*pvec)[i];
    222    }
    223    matrixElement->setAttributeNode(NewAttribute("values", pvalues.str()));
    224 
    225    defineElement->appendChild(matrixElement);
    226 }


Using an attribute for the values causes the truncation::

    136 xercesc::DOMAttr* G4GDMLWrite::NewAttribute(const G4String& name,
    137                                             const G4String& value)
    138 {
    139    xercesc::XMLString::transcode(name,tempStr,99);
    140    xercesc::DOMAttr* att = doc->createAttribute(tempStr);
    141    xercesc::XMLString::transcode(value,tempStr,99);
    142    att->setValue(tempStr);
    143    return att;
    144 }


Truncation is pushed out to 10,000 rather than 100 chars, in 10.5

/usr/local/opticks/externals/g4/geant4.10.05.b01/source/persistency/gdml/src/G4GDMLWriteMaterials.cc::

    210 void G4GDMLWriteMaterials::PropertyVectorWrite(const G4String& key,
    211                            const G4PhysicsOrderedFreeVector* const pvec)
    212 {
    213    const G4String matrixref = GenerateName(key, pvec);
    214    xercesc::DOMElement* matrixElement = NewElement("matrix");
    215    matrixElement->setAttributeNode(NewAttribute("name", matrixref));
    216    matrixElement->setAttributeNode(NewAttribute("coldim", "2"));
    217    std::ostringstream pvalues;
    218    for (size_t i=0; i<pvec->GetVectorLength(); i++)
    219    {
    220        if (i!=0)  { pvalues << " "; }
    221        pvalues << pvec->Energy(i) << " " << (*pvec)[i];
    222    }
    223    matrixElement->setAttributeNode(NewAttribute("values", pvalues.str()));
    224 
    225    defineElement->appendChild(matrixElement);
    226 }

    137 xercesc::DOMAttr* G4GDMLWrite::NewAttribute(const G4String& name,
    138                                             const G4String& value)
    139 {
    140    xercesc::XMLString::transcode(name,tempStr,9999);
    141    xercesc::DOMAttr* att = doc->createAttribute(tempStr);
    142    xercesc::XMLString::transcode(value,tempStr,9999);
    143    att->setValue(tempStr);
    144    return att;
    145 }



Github geant4 has no history, so cannot see when fixed.

* https://github.com/Geant4/geant4/blob/master/source/persistency/gdml/src/G4GDMLWrite.cc
* https://gitlab.cern.ch/geant4/geant4/commits/master/source/persistency/gdml/src/G4GDMLWrite.cc


Using blame on gitlab can see the fix was in 10.3.0 2 years ago.

* https://gitlab.cern.ch/geant4/geant4/blame/master/source/persistency/gdml/src/G4GDMLWrite.cc




