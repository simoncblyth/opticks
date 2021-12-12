#pragma once

#include <vector>
#include <string>

#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/util/XMLUni.hpp>
#include <xercesc/dom/DOM.hpp>

#include "CFG4_API_EXPORT.hh"

#include "plog/Severity.h"

struct Constant
{
    std::string                name ; 
    double                     value ;  
    xercesc::DOMElement*       constantElement ;
};

struct Matrix
{
    std::string                  name ; 
    std::string                  values ;  
    xercesc::DOMElement*         matrixElement ; 
    int                          repeat_index ;   

    bool operator==(const Matrix& rhs) const  { return name.compare(rhs.name) == 0 ; }
    std::string desc() const ; 

};



struct CFG4_API CGDMLKludgeRead
{
    static const plog::Severity LEVEL ; 

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
    std::vector<Matrix>  matrix ;  


    CGDMLKludgeRead( const char* path, bool kludge_truncated_matrix_); 
    virtual ~CGDMLKludgeRead(); 

    void DefineRead( const xercesc::DOMElement* const defineElement );
    void MatrixRead( const xercesc::DOMElement* const matrixElement, bool& truncated_values );
    Constant ConstantRead( const xercesc::DOMElement* const constantElement );

    int checkDuplicatedMatrix(); 
    int pruneDuplicatedMatrix(); 


    void KludgeTruncatedMatrix(xercesc::DOMElement* matrixElement );
    std::string KludgeFix( const char* values );


};



