#pragma once

#include "GDXML_API_EXPORT.hh"

struct GDXML_API GDXMLErrorHandler : public xercesc::ErrorHandler
{
    bool suppress ;

    GDXMLErrorHandler( bool suppress_ )
       :
       suppress(suppress_)
    {
    }

    void warning(const xercesc::SAXParseException& exception)
    {   
        if (suppress)  { return; }
        char* message = xercesc::XMLString::transcode(exception.getMessage());
        std::cout 
            << "GDXMLErrorHandler VALIDATION WARNING! " << message
            << " at line: " << exception.getLineNumber() 
            << std::endl
            ;
        xercesc::XMLString::release(&message);
    }   

    void error(const xercesc::SAXParseException& exception)
    {   
        if (suppress)  { return; }
        char* message = xercesc::XMLString::transcode(exception.getMessage());
        std::cout 
            << "GDXMLErrorHandler VALIDATION ERROR! " << message
            << " at line: " << exception.getLineNumber() 
            << std::endl
            ;
        xercesc::XMLString::release(&message);
    }   

    void fatalError(const xercesc::SAXParseException& exception)
    {   
        error(exception);
    }   
    void resetErrors() {}
};



