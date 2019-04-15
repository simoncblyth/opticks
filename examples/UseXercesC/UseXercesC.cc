

#include <xercesc/dom/DOM.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>


// 


int main(int argc, char** argv )
{

     xercesc::XMLPlatformUtils::Initialize();

     
     // G4GDMLWrite::Write   g4-;g4-cls G4GDMLWrite

      
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







     return 0 ; 
}

