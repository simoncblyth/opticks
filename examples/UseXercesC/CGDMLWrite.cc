#include <iostream>
#include <sstream>
#include "CGDMLWrite.hh"

CGDMLWrite::CGDMLWrite(xercesc::DOMDocument* doc_)
    :
    doc(doc_)
{
}

CGDMLWrite::~CGDMLWrite()
{
}

xercesc::DOMElement* CGDMLWrite::NewElement(const char* tagname)
{
   xercesc::XMLString::transcode(tagname,tempStr,9999);
   return doc->createElement(tempStr);
}


xercesc::DOMAttr* CGDMLWrite::NewAttribute(const char* name, const char* value)
{
   xercesc::XMLString::transcode(name,tempStr,9999);
   xercesc::DOMAttr* att = doc->createAttribute(tempStr);
   xercesc::XMLString::transcode(value,tempStr,9999);
   att->setValue(tempStr);
   return att;
}



/**
    In [3]: 1240./800./1e6
    Out[3]: 1.55e-06
    In [4]: 1240./80./1e6 
    Out[4]: 1.55e-05
**/

std::string CGDMLWrite::ConstantToMatrixValues(double value, double nm_lo, double nm_hi ) 
{
    double mev_lo = 1240./nm_hi/1e6 ; 
    double mev_hi = 1240./nm_lo/1e6 ; 

    std::stringstream ss ; 
    ss 
       << mev_lo << " " << value 
       << " "
       << mev_hi << " " << value 
       ; 

    std::string s = ss.str(); 
    return s ; 
}

/**
   <matrix coldim="2" name="bisMSBTIMECONSTANT0x6833de0" values="-1 1.4 1 1.4"/>
**/

xercesc::DOMElement* CGDMLWrite::ConstantToMatrixElement(const char* name, double value, double nm_lo, double nm_hi )
{
    std::string values = ConstantToMatrixValues(value, nm_lo, nm_hi ); 
    xercesc::DOMElement* matrixElement = NewElement("matrix");
    matrixElement->setAttributeNode(NewAttribute("name", name));
    matrixElement->setAttributeNode(NewAttribute("coldim", "2"));
    matrixElement->setAttributeNode(NewAttribute("values", values.c_str()));
    return matrixElement ; 
}


void CGDMLWrite::write(const char* path)
{
   std::cout << "CGDMLWrite::write " << path << std::endl ; 
   

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
      std::cout << "CGDMLWrite: Exception message is: " << message << std::endl;
      xercesc::XMLString::release(&message);
      return ;
   }
   catch (const xercesc::DOMException& toCatch)
   {
      char* message = xercesc::XMLString::transcode(toCatch.msg);
      std::cout << "CGDMLWrite: Exception message is: " << message << std::endl;
      xercesc::XMLString::release(&message);
      return ;
   }
   catch (...)
   {
      std::cout << "CGDMLWrite: Unexpected Exception!" << std::endl;
      return ;
   }

   delete myFormTarget;
   writer->release();



}
