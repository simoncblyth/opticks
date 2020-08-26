#pragma once

#include "plog/Severity.h"
#include "OXRAP_API_EXPORT.hh"
#include "OXPPNS.hh"

/**
OTexture
==========

* This is on the way out, replaced by OTex which uses OCtx 

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
};



 
