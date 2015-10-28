#pragma once

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>


class GScintillatorLib ;
template <typename T> class NPY ;

#include "OPropertyLib.hh"

class OScintillatorLib : public OPropertyLib {
    public:
        OScintillatorLib(optix::Context& ctx, GScintillatorLib* lib);
    public:
        void convert();
    private:
        void makeReemissionTexture(NPY<float>* buf);
    private:
        GScintillatorLib*    m_lib ;
};

inline OScintillatorLib::OScintillatorLib(optix::Context& ctx, GScintillatorLib* lib)
           : 
           OPropertyLib(ctx),
           m_lib(lib)
{
}








