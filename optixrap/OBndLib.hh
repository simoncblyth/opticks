#pragma once

#include "OXPPNS.hh"
#include <optixu/optixu_math_namespace.h>

class GBndLib ; 
class Opticks ; 
template <typename T> class NPY ;

#include "plog/Severity.h"
#include "OPropertyLib.hh"
#include "OXRAP_API_EXPORT.hh"

/**
OBndLib
=========

Translates and uploads into OptiX GPU context:

1. GBndLib NPY buffer into OptiX boundary_texture 
2. GBndLib NPY optical buffer into OptiX optical_buffer 

**/


class OXRAP_API OBndLib  : public OPropertyLib 
{
public:
    static const plog::Severity LEVEL ; 
public:
    OBndLib(optix::Context& ctx, GBndLib* lib);
public:
    unsigned getNumBnd();
    GBndLib* getBndLib(); 

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
    GBndLib*             m_blib ; 
    Opticks*             m_ok ; 
    NPY<float>*          m_debug_buffer ; 
    unsigned int         m_width ; 
    unsigned int         m_height ; 


};


