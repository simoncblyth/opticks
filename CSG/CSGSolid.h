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

Extract from notes in externals/optix7sdk.bash::

    CSGOptiX uses one GAS for each CSGSolid ("compound" of numPrim CSGPrim)
    and that one GAS always has only one buildInput which references
    numPrim SBT records which have "sbt-geometry-acceleration-structure-index" 
    of (0,1,2,...,numPrim-1)  


Saving/loading the vector of CSGSolid in CSGFoundry
is done by CSGFoundry::save,load,loadArray.
Due to this no pointers are used in the below
and the layout/alignment is kept simple for trivial 
debug access from python.  

**/

struct CSG_API CSGSolid   // Composite shape 
{
    char        label[16] ;   // sizeof 4 int 
    // raw use of so->label risks funny chars at slot 16, and truncation :  use getLabel() to avoid 

    int         numPrim ; 
    int         primOffset ;
    int         type = STANDARD_SOLID ;       

    char        intent = '\0' ;     
    char        pad0 ; 
    char        pad1 ; 
    char        pad2 ; 


    float4      center_extent ; 


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    static const plog::Severity LEVEL ;  

    const char* getLabel() const ; 
    bool labelMatch(const char* label) const ;  

    //char getLabelPrefix() const ; 
    char getIntent() const ;  // 'R' 'F' 'T'   used for forced triangulation
    void setIntent(char _intent); 


    static bool IsDiff( const CSGSolid& a , const CSGSolid& b ); 
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


