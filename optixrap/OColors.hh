#pragma once

#include "OXPPNS.hh"
#include <optixu/optixu_math_namespace.h>

class OpticksColors ; 
template <typename T> class NPY ; 


#include "OXRAP_API_EXPORT.hh"

class OXRAP_API OColors 
{
public:
    OColors(optix::Context& ctx, OpticksColors* colors);
public:
    void convert();
private:
#ifdef OLD_WAY
    optix::TextureSampler makeColorSampler(NPY<unsigned char>* colorBuffer);
    optix::TextureSampler makeSampler(NPY<unsigned char>* buffer, RTformat format, unsigned int nx, unsigned int ny);
#endif
private:
    optix::Context       m_context ; 
    OpticksColors*       m_colors ; 

};


