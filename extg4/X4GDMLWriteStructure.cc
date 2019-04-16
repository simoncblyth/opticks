#include "X4GDMLWriteStructure.hh"


X4GDMLWriteStructure::X4GDMLWriteStructure()
{
}

// /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/persistency/gdml/src/G4GDMLWrite.cc

void X4GDMLWriteStructure::write(const G4String& fname, const G4VSolid* solid )
{

   SchemaLocation = "SchemaLocation";
   addPointerToName = true ;


   xercesc::XMLString::transcode("LS", tempStr, 9999);
     xercesc::DOMImplementationRegistry::getDOMImplementation(tempStr);
   xercesc::XMLString::transcode("Range", tempStr, 9999);
   xercesc::DOMImplementation* impl =
     xercesc::DOMImplementationRegistry::getDOMImplementation(tempStr);
   xercesc::XMLString::transcode("gdml", tempStr, 9999);
   doc = impl->createDocument(0,tempStr,0);
   xercesc::DOMElement* gdml = doc->getDocumentElement();

#if XERCES_VERSION_MAJOR >= 3
                                             // DOM L3 as per Xerces 3.0 API
    xercesc::DOMLSSerializer* writer =
      ((xercesc::DOMImplementationLS*)impl)->createLSSerializer();

    xercesc::DOMConfiguration *dc = writer->getDomConfig();
    dc->setParameter(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true);

#else

   xercesc::DOMWriter* writer =
     ((xercesc::DOMImplementationLS*)impl)->createDOMWriter();

   if (writer->canSetFeature(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true))
       writer->setFeature(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true);

#endif

   gdml->setAttributeNode(NewAttribute("xmlns:xsi",
                          "http://www.w3.org/2001/XMLSchema-instance"));
   gdml->setAttributeNode(NewAttribute("xsi:noNamespaceSchemaLocation",
                          SchemaLocation));

   //ExtensionWrite(gdml);
   //DefineWrite(gdml);
   //MaterialsWrite(gdml);
   SolidsWrite(gdml);
   AddSolid( solid );

   //StructureWrite(gdml);
   //UserinfoWrite(gdml);
   //SetupWrite(gdml,logvol);

   //G4Transform3D R = TraverseVolumeTree(logvol,depth);

   //SurfacesWrite();
   xercesc::XMLFormatTarget *myFormTarget =
     new xercesc::LocalFileFormatTarget(fname.c_str());

   try
   {
#if XERCES_VERSION_MAJOR >= 3
                                            // DOM L3 as per Xerces 3.0 API
      xercesc::DOMLSOutput *theOutput =
        ((xercesc::DOMImplementationLS*)impl)->createLSOutput();
      theOutput->setByteStream(myFormTarget);
      writer->write(doc, theOutput);
#else
      writer->writeNode(myFormTarget, *doc);
#endif
   }
   catch (const xercesc::XMLException& toCatch)
   {
      char* message = xercesc::XMLString::transcode(toCatch.getMessage());
      G4cout << "G4GDML: Exception message is: " << message << G4endl;
      xercesc::XMLString::release(&message);
      //return G4Transform3D::Identity;
   }
   catch (const xercesc::DOMException& toCatch)
   {
      char* message = xercesc::XMLString::transcode(toCatch.msg);
      G4cout << "G4GDML: Exception message is: " << message << G4endl;
      xercesc::XMLString::release(&message);
      //return G4Transform3D::Identity;
   }
   catch (...)
   {   
      G4cout << "G4GDML: Unexpected Exception!" << G4endl;
      //return G4Transform3D::Identity;
   }        

   delete myFormTarget;
   writer->release();

}


