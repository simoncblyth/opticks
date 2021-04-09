#pragma once

#include <vector>
#include <string>

#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/util/XMLUni.hpp>
#include <xercesc/dom/DOM.hpp>

struct Constant
{
    std::string                name ; 
    double                     value ;  
    xercesc::DOMElement*       constantElement ;
};


struct CGDMLRead
{
    bool                      validate ; 
    bool                      kludge_truncated_matrix ; 

    xercesc::ErrorHandler*    handler ; 
    xercesc::XercesDOMParser* parser ; 
    xercesc::DOMDocument*     doc ; 
    xercesc::DOMElement*      element ;
    XMLCh                     tempStr[10000];

    xercesc::DOMElement*      the_defineElement = nullptr ;

    std::vector<Constant> constants ; 
    std::vector<xercesc::DOMElement*>  truncated_matrixElement ; 



    CGDMLRead( const char* path, bool kludge_truncated_matrix_); 
    virtual ~CGDMLRead(); 

    void DefineRead( const xercesc::DOMElement* const defineElement );
    void MatrixRead( const xercesc::DOMElement* const matrixElement, bool& truncated_values );
    Constant ConstantRead( const xercesc::DOMElement* const constantElement );

    void KludgeTruncatedMatrix(xercesc::DOMElement* matrixElement );
    std::string KludgeFix( const char* values );


};



