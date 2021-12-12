#pragma once

#include "CFG4_API_EXPORT.hh"

#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/util/XMLUni.hpp>
#include <xercesc/dom/DOM.hpp>

#include "plog/Severity.h"

struct CGDMLKludgeRead ; 
struct CGDMLKludgeWrite ; 

struct CFG4_API CGDMLKludge
{
    static const plog::Severity LEVEL ; 

    static const char* Fix(const char* srcpath); 

    CGDMLKludge(const char* srcpath) ; 
    virtual ~CGDMLKludge(); 

    const char*             srcpath ; 
    const char*             dstpath ; 

    bool                    kludge_truncated_matrix ;
    CGDMLKludgeRead*        reader ; 
    xercesc::DOMDocument*   doc  ; 
    xercesc::DOMElement*    defineElement ; 

    unsigned                num_duplicated_matrixElement ;  
    unsigned                num_pruned_matrixElement ;  

    unsigned                num_truncated_matrixElement ;  
    unsigned                num_constants ; 

    CGDMLKludgeWrite*       writer ;  
    bool                    issues ; 


    void replaceAllConstantWithMatrix(); 
};


