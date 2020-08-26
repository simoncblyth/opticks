#pragma once
#include "plog/Severity.h"
#include "OXRAP_API_EXPORT.hh"

/**
OTex
==========

* this is destined to replace OTexture 

Thinking to the future prefer interfaces not to "leak" the types of the 
API being wrapped. Aim for headers to not have any of the underlying types.

For example optix typedef enum types like RTtextureindexmode are cast to and 
from int to keep the interface clean.

**/

class NPYBase ; 

class OXRAP_API OTex  
{
    public:
        static const plog::Severity LEVEL ; 
        static void Upload2DLayeredTexture(const char* param_key, const char* domain_key, const NPYBase* inp, const char* config); 
    public:
        static const char* IndexModeString( int indexmode );
        static int         IndexMode( const char* config );
    private:
        static const char* INDEX_NORMALIZED_COORDINATES ; 
        static const char* INDEX_ARRAY_INDEX ; 

};



 
