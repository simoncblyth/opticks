#include "GGeoOptiXGeometry.hh"

#include <optixu/optixu_vector_types.h>

#include "RayTraceConfig.hh"
#include "GGeo.hh"
#include "GSolid.hh"
#include "GMesh.hh"

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
    m_gis.clear();

    GSolid* solid = m_ggeo->getSolid(0);

    traverseNode( solid, 0, true );

    printf("GGeoOptiXGeometry::convertStructure :  %lu gi \n", m_gis.size() );

    assert(m_gis.size() > 0);
}


void GGeoOptiXGeometry::traverseNode(GNode* node, unsigned int depth, bool recurse)
{
    GSolid* solid = dynamic_cast<GSolid*>(node) ;

    if(solid->isSelected())
    {
        optix::Geometry geometry = convertGeometry(solid) ;  
        addInstance(geometry, m_material );                 // tmp material override 
        m_ggeo->updateBounds(solid);
    }

    if(recurse)
    {
        for(unsigned int i = 0; i < node->getNumChildren(); i++) traverseNode(node->getChild(i), depth + 1, recurse);
    }
}


optix::Geometry GGeoOptiXGeometry::convertGeometry(GSolid* solid)
{
    optix::Geometry geometry = m_context->createGeometry();

    RayTraceConfig* cfg = RayTraceConfig::getInstance();
    geometry->setIntersectionProgram(cfg->createProgram("TriangleMesh.cu", "mesh_intersect"));
    geometry->setBoundingBoxProgram(cfg->createProgram("TriangleMesh.cu", "mesh_bounds"));

    GMatrixF* transform = solid->getTransform();
    GMesh* mesh = solid->getMesh();
    gfloat3* vertices = mesh->getTransformedVertices(*transform);

    unsigned int numVertices = mesh->getNumVertices();
    unsigned int numFaces = mesh->getNumFaces();

    geometry->setPrimitiveCount(numFaces);

    assert(sizeof(optix::float3) == sizeof(gfloat3)); 
    optix::Buffer vertexBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, numVertices);
    optix::float3* vertexBuffer_Host = static_cast<optix::float3*>( vertexBuffer->map() );
    geometry["vertexBuffer"]->setBuffer(vertexBuffer);
    memcpy( static_cast<void*>( vertexBuffer_Host ),
            static_cast<void*>( vertices ),
            sizeof( optix::float3 )*numVertices); 
    vertexBuffer->unmap();

    delete vertices ;


    assert(sizeof(optix::int3) == sizeof(guint3)); 
    optix::Buffer indexBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT3, numFaces );
    optix::int3* indexBuffer_Host = static_cast<optix::int3*>( indexBuffer->map() );
    geometry["indexBuffer"]->setBuffer(indexBuffer);
    memcpy( static_cast<void*>( indexBuffer_Host ),
            static_cast<void*>( mesh->getFaces() ),
            sizeof( optix::int3 )*numFaces); 
    indexBuffer->unmap();

    optix::Buffer emptyBuffer = m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, 0);
    geometry["tangentBuffer"]->setBuffer(emptyBuffer);
    geometry["bitangentBuffer"]->setBuffer(emptyBuffer);
    geometry["normalBuffer"]->setBuffer(emptyBuffer);
    geometry["texCoordBuffer"]->setBuffer(emptyBuffer);
 
    return geometry ; 
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
    gfloat3* p = m_ggeo->getLow();
    assert(p);
    return optix::make_float3(p->x, p->y, p->z); 
}

optix::float3 GGeoOptiXGeometry::getMax()
{
    gfloat3* p = m_ggeo->getHigh();
    assert(p);
    return optix::make_float3(p->x, p->y, p->z); 
}




