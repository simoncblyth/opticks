#include "GGeoOptiXGeometry.hh"

#include <optixu/optixu_vector_types.h>

#include "RayTraceConfig.hh"
#include "GGeo.hh"

//  analog to AssimpOptiXGeometry based on intermediary GGeo 

GGeoOptiXGeometry::~GGeoOptiXGeometry()
{
}

GGeoOptiXGeometry::GGeoOptiXGeometry(GGeo* ggeo)
           : 
           OptiXGeometry(),
           m_ggeo(ggeo)
{
}


void GGeoOptiXGeometry::convert()
{
    convertMaterials();
    convertStructure();
}


void GGeoOptiXGeometry::convertMaterials()
{
    for(unsigned int i=0 ; i < m_ggeo->getNumMaterials() ; i++ )
    {
        optix::Material material = convertMaterial(m_ggeo->getMaterial(i));
        m_materials.push_back(material);
    }
}


void GGeoOptiXGeometry::convertStructure()
{
}


optix::Material GGeoOptiXGeometry::convertMaterial(GMaterial* gmat)
{
    // NB material properties currently ignored

    RayTraceConfig* cfg = RayTraceConfig::getInstance();

    optix::Material material = m_context->createMaterial();

    material->setClosestHitProgram(0, cfg->createProgram("material1.cu", "closest_hit_radiance"));

    return material ; 
}




optix::float3 GGeoOptiXGeometry::getMin()
{
    aiVector3D* p = m_ageo->getLow();
    return optix::make_float3(p->x, p->y, p->z); 
}

optix::float3 GGeoOptiXGeometry::getMax()
{
    aiVector3D* p = m_ageo->getHigh();
    return optix::make_float3(p->x, p->y, p->z); 
}




