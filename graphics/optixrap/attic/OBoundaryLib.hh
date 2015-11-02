#pragma once

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_aabb_namespace.h>


class GBoundaryLib ; 
class GBuffer ; 

class OBoundaryLib  
{
public:
    OBoundaryLib(optix::Context& ctx, GBoundaryLib* lib);
public:
    void convert();
private:
    void convertBoundaryProperties(GBoundaryLib* blib);
    void convertColors(GBoundaryLib* blib);
private:
    optix::TextureSampler makeSampler(GBuffer* buffer, RTformat format, unsigned int nx, unsigned int ny);
    optix::TextureSampler makeWavelengthSampler(GBuffer* wavelengthBuffer);
    optix::TextureSampler makeColorSampler(GBuffer* colorBuffer);
    optix::float4         getDomain();
    optix::float4         getDomainReciprocal();
private:
    optix::Context       m_context ; 
    GBoundaryLib*        m_boundarylib ; 

};


inline OBoundaryLib::OBoundaryLib(optix::Context& ctx, GBoundaryLib* lib)
           : 
           m_context(ctx),
           m_boundarylib(lib)
{
}


