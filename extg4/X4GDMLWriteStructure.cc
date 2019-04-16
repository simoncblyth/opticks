#include "X4GDMLWriteStructure.hh"
#include <cstring>

#include <xercesc/framework/StdOutFormatTarget.hpp>
#include <xercesc/framework/MemBufFormatTarget.hpp>


X4GDMLWriteStructure::X4GDMLWriteStructure()
{
    init(); 
}

// /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/persistency/gdml/src/G4GDMLWrite.cc


void X4GDMLWriteStructure::write(const G4VSolid* solid, const char* path )
{
    add(solid); 
    write(path); 
}

std::string X4GDMLWriteStructure::to_string( const G4VSolid* solid )
{
    add(solid); 
    return write("MEMBUF") ; 
}





void X4GDMLWriteStructure::init()
{

   SchemaLocation = "SchemaLocation";
   addPointerToName = true ;

   xercesc::XMLString::transcode("LS", tempStr, 9999);
   xercesc::DOMImplementationRegistry::getDOMImplementation(tempStr);
   xercesc::XMLString::transcode("Range", tempStr, 9999);

   impl = xercesc::DOMImplementationRegistry::getDOMImplementation(tempStr);

   xercesc::XMLString::transcode("gdml", tempStr, 9999);
   doc = impl->createDocument(0,tempStr,0);

   gdml = doc->getDocumentElement();

   gdml->setAttributeNode(NewAttribute("xmlns:xsi",
                          "http://www.w3.org/2001/XMLSchema-instance"));
   gdml->setAttributeNode(NewAttribute("xsi:noNamespaceSchemaLocation",
                          SchemaLocation));

}





void X4GDMLWriteStructure::add(const G4VSolid* solid )
{
   SolidsWrite(gdml);
   AddSolid( solid );
}

std::string X4GDMLWriteStructure::write(const char* path)
{

#if XERCES_VERSION_MAJOR >= 3
                                             // DOM L3 as per Xerces 3.0 API
    xercesc::DOMLSSerializer* writer = ((xercesc::DOMImplementationLS*)impl)->createLSSerializer();

    xercesc::DOMConfiguration *dc = writer->getDomConfig();
    dc->setParameter(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true);

#else

   xercesc::DOMWriter* writer = ((xercesc::DOMImplementationLS*)impl)->createDOMWriter();

   if (writer->canSetFeature(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true))
       writer->setFeature(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true);

#endif


   xercesc::XMLFormatTarget* target = NULL ; 

   bool buf = false ; 

   if( path == NULL ) 
   {
       target = new xercesc::StdOutFormatTarget() ;
   }
   else if( strcmp(path,"MEMBUF") == 0 )
   {
       target = new xercesc::MemBufFormatTarget() ; 
       buf = true ; 
   }  
   else
   {
       target = new xercesc::LocalFileFormatTarget(path) ;
   }

   try
   {
#if XERCES_VERSION_MAJOR >= 3
                                            // DOM L3 as per Xerces 3.0 API
      xercesc::DOMLSOutput *theOutput = ((xercesc::DOMImplementationLS*)impl)->createLSOutput();
      theOutput->setByteStream(target);
      writer->write(doc, theOutput);
#else
      writer->writeNode(target, *doc);
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
   }
   catch (...)
   {   
      G4cout << "G4GDML: Unexpected Exception!" << G4endl;
   }        


   std::string ret ; 
   if(buf)
   {
       ret = (char*)((xercesc::MemBufFormatTarget*)target)->getRawBuffer() ; 
   }

   delete target;
   writer->release();

   return ret ; 
}








