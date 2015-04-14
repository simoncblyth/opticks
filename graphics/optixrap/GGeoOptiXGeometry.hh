#ifndef GGEOOPTIXGEOMETRY_H
#define GGEOOPTIXGEOMETRY_H

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_aabb_namespace.h>

#include "OptiXGeometry.hh"

class GGeo ; 
class GMergedMesh ; 
class GMaterial ; 
class GSolid ; 
class GNode ;
class GSubstance ;
class GPropertyMap ;
class GDrawable ; 

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

* NOT DONE : OptiX code to use substanceBuffer to return material/substance index 
  for the triangle intersected in intersection program 

* NOT DONE : flattening the substance library somehow ? eg into a collective
  substance buffer  
 
  Currently each GSubstance has 1-1 relationship with optix::Material
  which has one texture sampler. This is rather clean, the meat of 
  the sample is a wavelengthBuffer of float4 of dimension (39x4)
  39 from the number of wavelength samples and 4 from the number 
  of properties.

     material["wavelength_texture"]->setTextureSampler(sampler);

  Hmm its too soon to try to persist this kinda thing as its in flux.

*/

class GGeoOptiXGeometry  : public OptiXGeometry 
{
public:
    GGeoOptiXGeometry(GGeo* ggeo);
    GGeoOptiXGeometry(GGeo* ggeo, GMergedMesh* mergedmesh); // transitional 
    virtual ~GGeoOptiXGeometry();

public:
    void convert();

private:
    void convertSubstances();
    optix::Material convertSubstance(GSubstance* substance);
    void addWavelengthTexture(optix::Material& material, GPropertyMap* ptex);

private:
    void convertStructure();
    void traverseNode(GNode* node, unsigned int depth, bool recurse);

private:
    optix::Geometry convertGeometry(GSolid* solid);
    optix::GeometryInstance convertGeometryInstance(GSolid* solid);
private:
    // start migration to GMergedMesh
    optix::GeometryInstance convertDrawableInstance(GMergedMesh* mergedmesh);
    optix::Geometry convertDrawable(GMergedMesh* drawable);

public:
    optix::float3  getMin();
    optix::float3  getMax();
    optix::float3  getCenter();
    optix::float3  getExtent();
    optix::float3  getUp();

private:
    GGeo*        m_ggeo ; 
    GMergedMesh* m_mergedmesh ; 

};



#endif





