#pragma once

#include "CSGEnum.h"

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#include "plog/Severity.h"
#include <string>
#include "CSG_API_EXPORT.hh"
#endif


/**
CSGSolid
==========

Currently this is not uploaded to GPU, but still coding like 
it might be in future, ie keep it simple, no refs, 
128 bit (16 byte) alignment

**/

struct CSG_API CSGSolid   // Composite shape 
{


    char        label[16] ;   // sizeof 4 int 

    int         numPrim ; 
    int         primOffset ;
    int         type = STANDARD_SOLID ;       
    int         padding ;  

    float4      center_extent ; 


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    static const plog::Severity LEVEL ;  

    bool labelMatch(const char* label) const ;  

    static CSGSolid Make( const char* label_, int numPrim_, int primOffset_=-1 ); 
    static std::string MakeLabel(const char* typ0, unsigned idx0, char delim='_' );  
    static std::string MakeLabel(char typ0, unsigned idx0 );  
    static int         ParseLabel(const char* label, char& typ0, unsigned& idx0 );  
    int                get_ridx() const ; 
 
    static std::string MakeLabel(char typ0, unsigned idx0, char typ1, unsigned idx1 );  
    static std::string MakeLabel(char typ0, unsigned idx0, char typ1, unsigned idx1, char typ2, unsigned idx2 );  
    std::string desc() const ; 
#endif

};


