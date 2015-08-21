#pragma once

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_aabb_namespace.h>

#include "OptiXGeometry.hh"

/*

Migration to the flattened GMergedMesh 
=======================================

Motivation:
-----------

* single geometry instance in recommended as fastest OptiX geometry handling technique
* potentially can persist the buffers rather easily and use them as a geocache
  to avoid parsing at initialization, enabling a fast start 

Progress:
----------

* geometry creation DONE : using convertDrawableInstance and convertDrawable to 
  pump the flattened GMergedMesh to OptiX using a single geometry instance
  with multiple materials

* IN PROGRESS : flattening the substance library into the WavelengthBuffer
 
  Formerly each GSubstance has 1-1 relationship with optix::Material
  which has one texture sampler. Although clean, its kinda pointless
  having order 58 materials which only differ by the textures they contain.

  Trying to move to single material operation with properties using 
  the substanceBuffer to get the substanceIndex which is used
  for wavelength lookups into the waveLengthBuffer.

     material["wavelength_texture"]->setTextureSampler(sampler);

*/


class GGeo ; 
class GMergedMesh ; 
class GBoundaryLib ; 
class GBuffer ; 

// canonical usage from OptiXEngine::initGeometry
//
// TODO: rename to ?MeshGeometry? as handle multiple GMergedMesh 
//       with instancing support 
//

class OGeo  : public OptiXGeometry 
{
public:
    OGeo(GGeo* gg, GBoundaryLib* lib);

public:
    void convert();

public:
    // hmm maybe split blib handling into separate class ?
    void                  convertBoundaryProperties(GBoundaryLib* blib);
    optix::TextureSampler makeWavelengthSampler(GBuffer* wavelengthBuffer);
    optix::TextureSampler makeReemissionSampler(GBuffer* reemissionBuffer);
    optix::float4         getDomain();
    optix::float4         getDomainReciprocal();

public:
    template <typename T>
    optix::Buffer createInputBuffer(GBuffer* buf, RTformat format, unsigned int fold=1);

private:
    optix::GeometryInstance makeGeometryInstance(GMergedMesh* mergedmesh);
    optix::Geometry         makeGeometry(GMergedMesh* mergedmesh);

public:
    optix::float3  getMin();
    optix::float3  getMax();

private:
    GGeo*         m_ggeo ; 
    GBoundaryLib* m_boundarylib ; 

};


inline OGeo::OGeo(GGeo* gg, GBoundaryLib* lib)
           : 
           OptiXGeometry(),
           m_ggeo(gg),
           m_boundarylib(lib)
{
}


