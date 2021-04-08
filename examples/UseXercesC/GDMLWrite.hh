#pragma once

#include <xercesc/dom/DOM.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>

struct GDMLWrite 
{
    GDMLWrite(const xercesc::DOMDocument* doc); 
    void write(const char* path);

    virtual ~GDMLWrite(); 

    const xercesc::DOMDocument*   doc;
    xercesc::DOMElement*    extElement;
    xercesc::DOMElement*    userinfoElement;
    XMLCh                   tempStr[10000];

};
