#pragma once


struct CGDMLKludgeErrorHandler : public xercesc::ErrorHandler
{
    bool suppress ;

    CGDMLKludgeErrorHandler( bool suppress_ )
       :
       suppress(suppress_)
    {
    }

    void warning(const xercesc::SAXParseException& exception)
    {   
        if (suppress)  { return; }
        char* message = xercesc::XMLString::transcode(exception.getMessage());
        std::cout 
            << "CGDMLKludgeErrorHandler VALIDATION WARNING! " << message
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
            << "CGDMLKludgeErrorHandler VALIDATION ERROR! " << message
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



