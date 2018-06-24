#pragma once

#include "OXPPNS.hh"
#include <optixu/optixu_math_namespace.h>


class GSourceLib ;
template <typename T> class NPY ;

#include "OPropertyLib.hh"
#include "OXRAP_API_EXPORT.hh"

/**
OSourceLib
===========

Converts the GSourceLib buffer into a texture that 
is accessible in the OptiX context GPU side using::

   source_texture
   source_domain

**/

class OXRAP_API OSourceLib : public OPropertyLib {
    public:
        OSourceLib(optix::Context& ctx, GSourceLib* lib);
    public:
        void convert();
    private:
        void makeSourceTexture(NPY<float>* buf);
    private:
        GSourceLib*    m_lib ;
};







