#include <iostream>
#include "GDMLWrite.hh"

GDMLWrite::GDMLWrite(const xercesc::DOMDocument* doc_)
    :
    doc(doc_)
{
}

GDMLWrite::~GDMLWrite()
{
}


void GDMLWrite::write(const char* path)
{
   std::cout << "GDMLWrite::write " << path << std::endl ; 
   

   xercesc::XMLString::transcode("LS", tempStr, 9999);
   xercesc::DOMImplementationRegistry::getDOMImplementation(tempStr);
   xercesc::XMLString::transcode("Range", tempStr, 9999);
   xercesc::DOMImplementation* impl = xercesc::DOMImplementationRegistry::getDOMImplementation(tempStr);

   if( doc == nullptr )
   { 
       xercesc::XMLString::transcode("gdml", tempStr, 9999);
       doc = impl->createDocument(0,tempStr,0);
       //xercesc::DOMElement* gdml = doc->getDocumentElement();
   }

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



   xercesc::XMLFormatTarget *myFormTarget = new xercesc::LocalFileFormatTarget(path);

   try
   {
#if XERCES_VERSION_MAJOR >= 3
                                            // DOM L3 as per Xerces 3.0 API
      xercesc::DOMLSOutput *theOutput = ((xercesc::DOMImplementationLS*)impl)->createLSOutput(); 
      theOutput->setByteStream(myFormTarget);
      writer->write(doc, theOutput);
#else
      writer->writeNode(myFormTarget, *doc);
#endif
   }
   catch (const xercesc::XMLException& toCatch)
   {
      char* message = xercesc::XMLString::transcode(toCatch.getMessage());
      std::cout << "GDMLWrite: Exception message is: " << message << std::endl;
      xercesc::XMLString::release(&message);
      return ;
   }
   catch (const xercesc::DOMException& toCatch)
   {
      char* message = xercesc::XMLString::transcode(toCatch.msg);
      std::cout << "GDMLWrite: Exception message is: " << message << std::endl;
      xercesc::XMLString::release(&message);
      return ;
   }
   catch (...)
   {
      std::cout << "GDMLWrite: Unexpected Exception!" << std::endl;
      return ;
   }

   delete myFormTarget;
   writer->release();



}
