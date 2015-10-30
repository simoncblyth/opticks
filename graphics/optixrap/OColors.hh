#pragma once

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>

class GColors ; 
class GBuffer ; 

// TODO: avoid duplication of makeSampler with OPropertyLib by moving it to OContext and using that

class OColors 
{
public:
    OColors(optix::Context& ctx, GColors* colors);
public:
    void convert();
private:
    optix::TextureSampler makeColorSampler(GBuffer* colorBuffer);
    optix::TextureSampler makeSampler(GBuffer* buffer, RTformat format, unsigned int nx, unsigned int ny);
private:
    optix::Context       m_context ; 
    GColors*             m_colors ; 

};


inline OColors::OColors(optix::Context& ctx, GColors* colors)
           : 
           m_context(ctx),
           m_colors(colors)
{
}


