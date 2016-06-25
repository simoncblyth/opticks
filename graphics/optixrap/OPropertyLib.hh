#pragma once

#include <optixu/optixpp_namespace.h>
template <typename T> class NPY ;

#include "OXRAP_API_EXPORT.hh"

class OXRAP_API OPropertyLib  {
    public:
        OPropertyLib(optix::Context& ctx);
    public:
/*
        optix::TextureSampler makeTexture(NPY<float>* buffer, RTformat format, unsigned int nx, unsigned int ny, bool empty=false);
*/
    protected:
        void upload(optix::Buffer& optixBuffer, NPY<float>* buffer);
        void configureSampler(optix::TextureSampler& sampler, optix::Buffer& buffer);
        void dumpVals( float* vals, unsigned int n);
    protected:
        optix::Context       m_context ; 

};


