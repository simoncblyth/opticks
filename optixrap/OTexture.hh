#pragma once

#include "OXPPNS.hh"
#include "plog/Severity.h"
#include "OXRAP_API_EXPORT.hh"

/**
OTexture
==========


Thinking to the future prefer interfaces not to "leak" the types of the 
API being wrapped. Aim for headers to not have any of the underlying types.

For example optix typedef enum types like RTtextureindexmode are cast to and 
from int to keep the interface clean.

TODO: avoid the optix::Context 


**/

class NPYBase ; 

class OXRAP_API OTexture  
{
    public:
        static const plog::Severity LEVEL ; 
        template <typename T> static void Upload2DLayeredTexture(optix::Context& context, const char* param_key, const char* domain_key, const NPYBase* inp, const char* config); 

    public:

        static const char* IndexModeString( int indexmode );
        static int         IndexMode( const char* config );
    private:
        static const char* INDEX_NORMALIZED_COORDINATES ; 
        static const char* INDEX_ARRAY_INDEX ; 

};



 
