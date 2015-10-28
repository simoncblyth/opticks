#pragma once

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>


class GBndLib ; 
template <typename T> class NPY ;

#include "OPropertyLib.hh"

class OBndLib  : public OPropertyLib 
{
public:
    OBndLib(optix::Context& ctx, GBndLib* lib);
public:
    void convert();
private:
    void makeBoundaryTexture(NPY<float>* buf);
private:
    GBndLib*             m_lib ; 

};


inline OBndLib::OBndLib(optix::Context& ctx, GBndLib* lib)
           : 
           OPropertyLib(ctx),
           m_lib(lib)
{
}


