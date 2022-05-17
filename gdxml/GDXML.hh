#pragma once
/**
GDXML.hh
================



**/
#include "GDXML_API_EXPORT.hh"

#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/util/XMLUni.hpp>
#include <xercesc/dom/DOM.hpp>

#include "plog/Severity.h"

struct GDXMLRead ; 
struct GDXMLWrite ; 

struct GDXML_API GDXML
{
    static const plog::Severity LEVEL ; 

    static const char* Fix(const char* srcpath); 

    GDXML(const char* srcpath) ; 
    virtual ~GDXML(); 

    const char*             srcpath ; 
    const char*             dstpath ; 

    bool                    kludge_truncated_matrix ;
    GDXMLRead*              reader ; 
    xercesc::DOMDocument*   doc  ; 
    xercesc::DOMElement*    defineElement ; 

    unsigned                num_duplicated_matrixElement ;  
    unsigned                num_pruned_matrixElement ;  

    unsigned                num_truncated_matrixElement ;  
    unsigned                num_constants ; 

    GDXMLWrite*             writer ;  
    bool                    issues ; 


    void replaceAllConstantWithMatrix(); 
};


