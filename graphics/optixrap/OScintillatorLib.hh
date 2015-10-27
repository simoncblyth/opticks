#pragma once

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_aabb_namespace.h>


class GScintillatorLib ;
template <typename T> class NPY ;


class OScintillatorLib {
    public:
        OScintillatorLib(optix::Context& ctx, GScintillatorLib* lib);
    public:
        void convert();
    private:
        void makeReemissionTexture(NPY<float>* buf);
        optix::TextureSampler makeTexture(NPY<float>* buffer, RTformat format, unsigned int nx, unsigned int ny);
    private:
        optix::Context       m_context ; 
        GScintillatorLib*    m_lib ;
};

inline OScintillatorLib::OScintillatorLib(optix::Context& ctx, GScintillatorLib* lib)
           : 
           m_context(ctx),
           m_lib(lib)
{
}








