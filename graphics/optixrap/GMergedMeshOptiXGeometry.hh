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


class GMergedMesh ; 
class GBuffer ; 

class GMergedMeshOptiXGeometry  : public OptiXGeometry 
{
public:
    GMergedMeshOptiXGeometry(GMergedMesh* mergedmesh); 

public:
    void convert();
    optix::TextureSampler makeTextureSampler(GBuffer* wbuf);
    optix::float3 getDomain();
    optix::float3 getDomainReciprocal();

private:
    optix::GeometryInstance convertDrawableInstance(GMergedMesh* mergedmesh);
    optix::Geometry convertDrawable(GMergedMesh* drawable);
    optix::Material makeMaterial(GBuffer* wbuf);

public:
    optix::float3  getMin();
    optix::float3  getMax();
/*
    optix::float3  getCenter();
    optix::float3  getExtent();
    optix::float3  getUp();
*/

private:
    GMergedMesh* m_mergedmesh ; 

};



