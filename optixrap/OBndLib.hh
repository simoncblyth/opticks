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
    unsigned getNumBnd();

    void setDebugBuffer(NPY<float>* npy);

    void setWidth(unsigned int width);
    void setHeight(unsigned int height);
    unsigned getWidth();
    unsigned getHeight();

    void convert();
private:
    void makeBoundaryTexture(NPY<float>* buf);
    void makeBoundaryOptical(NPY<unsigned int>* obuf);
private:
    GBndLib*             m_lib ; 
    NPY<float>*          m_debug_buffer ; 
    unsigned int         m_width ; 
    unsigned int         m_height ; 


};


