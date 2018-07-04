#pragma once

#include "OXPPNS.hh"
template <typename T> class NPY ;

#include "OXRAP_API_EXPORT.hh"

class OXRAP_API OPropertyLib  {
    public:
        OPropertyLib(optix::Context& ctx, const char* name);
    public:
    protected:
        void upload(optix::Buffer& optixBuffer, NPY<float>* buffer);
        void dumpVals( float* vals, unsigned int n);
    protected:
        optix::Context       m_context ; 
        const char*          m_name ; 

};


