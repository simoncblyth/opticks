#pragma once

#include <optixu/optixpp_namespace.h>
template <typename T> class NPY ;

class OPropertyLib  {
    public:
        OPropertyLib(optix::Context& ctx);
    public:
        optix::TextureSampler makeTexture(NPY<float>* buffer, RTformat format, unsigned int nx, unsigned int ny);
    protected:
        optix::Context       m_context ; 

};

inline OPropertyLib::OPropertyLib(optix::Context& ctx) : m_context(ctx)
{
}



