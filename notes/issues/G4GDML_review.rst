G4GDML_review
==================

Overview
---------

Context :doc:`torus_replacement_on_the_fly`

Review how GDML writing works with a view to creation of x4 X4GDMLWrite 
that can write GDML snippets for single solids.

See also

* cmake/Modules/FindOpticksXercesC
* examples/UseXercesC
* examples/UseOpticksXercesC


g4-;g4-cls G4GDMLWriteSolids
---------------------------------

::

    79 class G4GDMLWriteSolids : public G4GDMLWriteMaterials
    80 {
    ...
    99 
    100   public:
    101 
    102    virtual void AddSolid(const G4VSolid* const);
    103    virtual void SolidsWrite(xercesc::DOMElement*);
    104 
    105   protected:
    106 
    107    G4GDMLWriteSolids();
    108    virtual ~G4GDMLWriteSolids();
    109 
    110    void MultiUnionWrite(xercesc::DOMElement* solElement, const G4MultiUnion* const);
    111    void BooleanWrite(xercesc::DOMElement*, const G4BooleanSolid* const);
    112    void ScaledWrite(xercesc::DOMElement*, const G4ScaledSolid* const);
    113    void BoxWrite(xercesc::DOMElement*, const G4Box* const);
    114    void ConeWrite(xercesc::DOMElement*, const G4Cons* const);
    115    void ElconeWrite(xercesc::DOMElement*, const G4EllipticalCone* const);
    116    void EllipsoidWrite(xercesc::DOMElement*, const G4Ellipsoid* const);
    117    void EltubeWrite(xercesc::DOMElement*, const G4EllipticalTube* const);


    1110 void G4GDMLWriteSolids::SolidsWrite(xercesc::DOMElement* gdmlElement)
    1111 {
    1112 #ifdef G4VERBOSE
    1113    G4cout << "G4GDML: Writing solids..." << G4endl;
    1114 #endif
    1115    solidsElement = NewElement("solids");
    1116    gdmlElement->appendChild(solidsElement);
    1117 
    1118    solidList.clear();
    1119 }


    1121 void G4GDMLWriteSolids::AddSolid(const G4VSolid* const solidPtr)
    1122 {
    1123    for (size_t i=0; i<solidList.size(); i++)   // Check if solid is
    1124    {                                           // already in the list!
    1125       if (solidList[i] == solidPtr)  { return; }
    1126    }
    1127 
    1128    solidList.push_back(solidPtr);
    1129 
    1130    if (const G4BooleanSolid* const booleanPtr
    1131      = dynamic_cast<const G4BooleanSolid*>(solidPtr))
    1132      { BooleanWrite(solidsElement,booleanPtr); } else
    1133    if (const G4ScaledSolid* const scaledPtr
    1134      = dynamic_cast<const G4ScaledSolid*>(solidPtr))
    1135      { ScaledWrite(solidsElement,scaledPtr); } else
    1136    if (solidPtr->GetEntityType()=="G4MultiUnion")
    1137      { const G4MultiUnion* const munionPtr
    1138      = static_cast<const G4MultiUnion*>(solidPtr);
    1139        MultiUnionWrite(solidsElement,munionPtr); } else
    1140    if (solidPtr->GetEntityType()=="G4Box")




::

    167 G4Transform3D G4GDMLWrite::Write(const G4String& fname,
    168                                  const G4LogicalVolume* const logvol,
    169                                  const G4String& setSchemaLocation,
    170                                  const G4int depth,
    171                                        G4bool refs)
    172 {
    173    SchemaLocation = setSchemaLocation;
    174    addPointerToName = refs;
    175 #ifdef G4VERBOSE
    176    if (depth==0) { G4cout << "G4GDML: Writing '" << fname << "'..." << G4endl; }
    177    else   { G4cout << "G4GDML: Writing module '" << fname << "'..." << G4endl; }
    178 #endif
    179    if (FileExists(fname))
    180    {
    181      G4String ErrorMessage = "File '"+fname+"' already exists!";
    182      G4Exception("G4GDMLWrite::Write()", "InvalidSetup",
    183                  FatalException, ErrorMessage);
    184    }
    185   
    186    VolumeMap().clear(); // The module map is global for all modules,
    187                         // so clear it only at once!
    188 
    189    xercesc::XMLString::transcode("LS", tempStr, 9999);
    190      xercesc::DOMImplementationRegistry::getDOMImplementation(tempStr);
    191    xercesc::XMLString::transcode("Range", tempStr, 9999);
    192    xercesc::DOMImplementation* impl =
    193      xercesc::DOMImplementationRegistry::getDOMImplementation(tempStr);
    194    xercesc::XMLString::transcode("gdml", tempStr, 9999);
    195    doc = impl->createDocument(0,tempStr,0);
    196    xercesc::DOMElement* gdml = doc->getDocumentElement();
    197 
    198 #if XERCES_VERSION_MAJOR >= 3
    199                                              // DOM L3 as per Xerces 3.0 API
    200     xercesc::DOMLSSerializer* writer =
    201       ((xercesc::DOMImplementationLS*)impl)->createLSSerializer();
    202 
    203     xercesc::DOMConfiguration *dc = writer->getDomConfig();
    204     dc->setParameter(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true);
    205 
    206 #else
    207 
    208    xercesc::DOMWriter* writer =
    209      ((xercesc::DOMImplementationLS*)impl)->createDOMWriter();
    210 
    211    if (writer->canSetFeature(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true))
    212        writer->setFeature(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true);
    213 
    214 #endif
    215 
    216    gdml->setAttributeNode(NewAttribute("xmlns:xsi",
    217                           "http://www.w3.org/2001/XMLSchema-instance"));
    218    gdml->setAttributeNode(NewAttribute("xsi:noNamespaceSchemaLocation",
    219                           SchemaLocation));
    220 
    221    ExtensionWrite(gdml);
    222    DefineWrite(gdml);
    223    MaterialsWrite(gdml);
    224    SolidsWrite(gdml);
    225    StructureWrite(gdml);
    226    UserinfoWrite(gdml);




G4GDML setup code
----------------------

::

    blyth@localhost src]$ grep xerces *.cc | grep -v DOMElement | grep -v XMLString | grep -v DOMNode | grep -v DOMNamed | grep -v DOMAttr 
    G4GDMLParser.cc:  xercesc::XMLPlatformUtils::Initialize();
    G4GDMLParser.cc:  xercesc::XMLPlatformUtils::Initialize();
    G4GDMLParser.cc:  xercesc::XMLPlatformUtils::Initialize();
    G4GDMLParser.cc:  xercesc::XMLPlatformUtils::Terminate();
    G4GDMLRead.cc:   xercesc::ErrorHandler* handler = new G4GDMLErrorHandler(!validate);
    G4GDMLRead.cc:   xercesc::XercesDOMParser* parser = new xercesc::XercesDOMParser;
    G4GDMLRead.cc:     parser->setValidationScheme(xercesc::XercesDOMParser::Val_Always);
    G4GDMLRead.cc:   catch (const xercesc::XMLException &e)
    G4GDMLRead.cc:   catch (const xercesc::DOMException &e)
    G4GDMLRead.cc:   xercesc::DOMDocument* doc = parser->getDocument();
    G4GDMLWrite.cc:     xercesc::DOMImplementationRegistry::getDOMImplementation(tempStr);
    G4GDMLWrite.cc:   xercesc::DOMImplementation* impl =
    G4GDMLWrite.cc:     xercesc::DOMImplementationRegistry::getDOMImplementation(tempStr);
    G4GDMLWrite.cc:    xercesc::DOMLSSerializer* writer =
    G4GDMLWrite.cc:      ((xercesc::DOMImplementationLS*)impl)->createLSSerializer();
    G4GDMLWrite.cc:    xercesc::DOMConfiguration *dc = writer->getDomConfig();
    G4GDMLWrite.cc:    dc->setParameter(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true);
    G4GDMLWrite.cc:   xercesc::DOMWriter* writer =
    G4GDMLWrite.cc:     ((xercesc::DOMImplementationLS*)impl)->createDOMWriter();
    G4GDMLWrite.cc:   if (writer->canSetFeature(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true))
    G4GDMLWrite.cc:       writer->setFeature(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true);
    G4GDMLWrite.cc:   xercesc::XMLFormatTarget *myFormTarget =
    G4GDMLWrite.cc:     new xercesc::LocalFileFormatTarget(fname.c_str());
    G4GDMLWrite.cc:      xercesc::DOMLSOutput *theOutput =
    G4GDMLWrite.cc:        ((xercesc::DOMImplementationLS*)impl)->createLSOutput();
    G4GDMLWrite.cc:   catch (const xercesc::XMLException& toCatch)
    G4GDMLWrite.cc:   catch (const xercesc::DOMException& toCatch)
    [blyth@localhost src]$ 




