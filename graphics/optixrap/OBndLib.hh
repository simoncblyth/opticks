#pragma once

#include "OXPPNS.hh"
#include <optixu/optixu_math_namespace.h>

class GBndLib ; 
template <typename T> class NPY ;

#include "OPropertyLib.hh"
#include "OXRAP_API_EXPORT.hh"

class OXRAP_API OBndLib  : public OPropertyLib 
{
public:
    OBndLib(optix::Context& ctx, GBndLib* lib);
public:
    void convert();
private:
    void makeBoundaryTexture(NPY<float>* buf);
    void makeBoundaryOptical(NPY<unsigned int>* obuf);
private:
    GBndLib*             m_lib ; 

};


