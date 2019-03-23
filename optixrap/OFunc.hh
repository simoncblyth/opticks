#pragma once
/**
OFunc
=======

OptiX callable program handling.

**/


#include "OXPPNS.hh"
#include <optixu/optixu_math_namespace.h>

#include "OXRAP_API_EXPORT.hh"

class OContext ; 

class OXRAP_API OFunc
{
public:
    OFunc(OContext* ocontext, const char* ptxname, const char* ctxname, const char* funcnames);
    void convert();
private:
    OContext*            m_ocontext ; 
    optix::Context       m_context ; 

    const char*   m_ptxname ; 
    const char*   m_ctxname ; 
    const char*   m_funcnames ; 
 
};


