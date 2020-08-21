#pragma once

#include "OXPPNS.hh"
#include "plog/Severity.h"
#include "OXRAP_API_EXPORT.hh"

/**
OTexture
==========

**/

class NPYBase ; 

class OXRAP_API OTexture  
{
    public:
        static const plog::Severity LEVEL ; 
        template <typename T> static void Upload2DLayeredTexture(optix::Context& context, const char* param_key, const char* domain_key, const NPYBase* inp); 
};



 
