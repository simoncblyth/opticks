#pragma once

#include <vector>
#include <string>

#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/util/XMLUni.hpp>
#include <xercesc/dom/DOM.hpp>

struct GDMLRead
{
    bool                      validate ; 
    bool                      kludge_truncated_matrix ; 

    xercesc::ErrorHandler*    handler ; 
    xercesc::XercesDOMParser* parser ; 
    xercesc::DOMDocument*     doc ; 
    xercesc::DOMElement*      element ;
    XMLCh                     tempStr[10000];


    std::vector<xercesc::DOMElement*>  truncated_matrixElement ; 

    GDMLRead( const char* path, bool kludge_truncated_matrix_); 
    virtual ~GDMLRead(); 

    void DefineRead( const xercesc::DOMElement* const defineElement );
    void MatrixRead( const xercesc::DOMElement* const matrixElement, bool& truncated_values );
    void ConstantRead( const xercesc::DOMElement* const constantElement );

    void KludgeTruncatedMatrix(xercesc::DOMElement* matrixElement );
    std::string KludgeFix( const char* values );


};



