#pragma once

struct GDMLRead ; 
struct GDMLWrite ; 

struct GDMLKludgeFix
{
    GDMLKludgeFix(const char* srcpath) ; 
    virtual ~GDMLKludgeFix(); 

    const char*             srcpath ; 
    const char*             dstpath ; 

    bool                    kludge_truncated_matrix ;
    GDMLRead*               reader ; 
    xercesc::DOMDocument*   doc  ; 
    xercesc::DOMElement*    defineElement ; 

    unsigned                num_truncated_matrixElement ;  
    unsigned                num_constants ; 

    GDMLWrite*              writer ;  


    void replaceAllConstantWithMatrix(); 
};


