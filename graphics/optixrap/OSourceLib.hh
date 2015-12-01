#pragma once

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>


class GSourceLib ;
template <typename T> class NPY ;

#include "OPropertyLib.hh"

class OSourceLib : public OPropertyLib {
    public:
        OSourceLib(optix::Context& ctx, GSourceLib* lib);
    public:
        void convert();
    private:
        void makeSourceTexture(NPY<float>* buf);
    private:
        GSourceLib*    m_lib ;
};

inline OSourceLib::OSourceLib(optix::Context& ctx, GSourceLib* lib)
           : 
           OPropertyLib(ctx),
           m_lib(lib)
{
}








