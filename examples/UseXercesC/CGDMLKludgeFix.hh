#pragma once

struct CGDMLRead ; 
struct CGDMLWrite ; 

struct CGDMLKludgeFix
{
    CGDMLKludgeFix(const char* srcpath) ; 
    virtual ~CGDMLKludgeFix(); 

    const char*             srcpath ; 
    const char*             dstpath ; 

    bool                    kludge_truncated_matrix ;
    CGDMLRead*              reader ; 
    xercesc::DOMDocument*   doc  ; 
    xercesc::DOMElement*    defineElement ; 

    unsigned                num_truncated_matrixElement ;  
    unsigned                num_constants ; 

    CGDMLWrite*             writer ;  


    void replaceAllConstantWithMatrix(); 
};


