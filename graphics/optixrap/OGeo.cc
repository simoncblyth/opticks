#include "OGeo.hh"
#include "OEngine.hh"

#include "GGeo.hh"
#include "GMergedMesh.hh"
#include "GBoundaryLib.hh"


// npy-
#include "stringutil.hpp"


#include "RayTraceConfig.hh"

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal



void OGeo::setGeometryGroup(optix::GeometryGroup ggrp)
{
    m_geometry_group = ggrp ; 
}

void OGeo::convert()
{
    unsigned int nmm = m_ggeo->getNumMergedMesh();
    LOG(info) << "OGeo::convert"
              << " nmm " << nmm
              ;

    for(unsigned int i=0 ; i < nmm ; i++)
    {
        GMergedMesh* mm = m_ggeo->getMergedMesh(i);
        assert(mm);

        optix::GeometryInstance gi = makeGeometryInstance(mm);
        m_gis.push_back(gi);
    }
}

void OGeo::setupAcceleration()
{
    const char* builder = "Sbvh" ;
    const char* traverser = "Bvh" ;

    LOG(info) << "OGeo::setupAcceleration for " 
              << " gis " <<  m_gis.size() 
              << " builder " << builder 
              << " traverser " << traverser
              ; 
    
    optix::Acceleration acceleration = m_context->createAcceleration(builder, traverser);
    acceleration->setProperty( "vertex_buffer_name", "vertexBuffer" );
    acceleration->setProperty( "index_buffer_name", "indexBuffer" );

    m_geometry_group->setAcceleration( acceleration );

    acceleration->markDirty();

    m_geometry_group->setChildCount(m_gis.size());
    for(unsigned int i=0 ; i <m_gis.size() ; i++) m_geometry_group->setChild(i, m_gis[i]);

    // FOR UNKNOWN REASONS SETTING top_object CAUSES SEGFAULT WHEN USED FROM SEPARATE SO 
    // AND NOT WHEN ALL COMPILED INTO SAME EXECUTABLE
    // ... IT DUPLICATES A SETTING IN MeshViewer ANYHOW SO NO PROBLEM SKIPPING IT 
    //
    //  assuming a not-updated lib is the cause
    //
    //m_context["top_object"]->set(m_geometry_group);

    LOG(info) << "OGeo::setupAcceleration DONE ";
}





optix::GeometryInstance OGeo::makeGeometryInstance(GMergedMesh* mergedmesh)
{
    LOG(info) << "OGeo::makeGeometryInstance material1  " ; 

    optix::Material material = m_context->createMaterial();
    RayTraceConfig* cfg = RayTraceConfig::getInstance();
    material->setClosestHitProgram(OEngine::e_radiance_ray, cfg->createProgram("material1_radiance.cu", "closest_hit_radiance"));
    material->setClosestHitProgram(OEngine::e_propagate_ray, cfg->createProgram("material1_propagate.cu", "closest_hit_propagate"));

    std::vector<optix::Material> materials ;
    materials.push_back(material);

    optix::Geometry geometry = makeGeometry(mergedmesh) ;  
    optix::GeometryInstance gi = m_context->createGeometryInstance( geometry, materials.begin(), materials.end()  );  

    return gi ;
}


optix::Geometry OGeo::makeGeometry(GMergedMesh* mergedmesh)
{
    // index buffer items are the indices of every triangle vertex, so divide by 3 to get faces 
    // and use folding by 3 in createInputBuffer

    optix::Geometry geometry = m_context->createGeometry();
    RayTraceConfig* cfg = RayTraceConfig::getInstance();
    geometry->setIntersectionProgram(cfg->createProgram("TriangleMesh.cu", "mesh_intersect"));
    geometry->setBoundingBoxProgram(cfg->createProgram("TriangleMesh.cu", "mesh_bounds"));

    GBuffer* vbuf = mergedmesh->getVerticesBuffer();
    GBuffer* ibuf = mergedmesh->getIndicesBuffer();
    GBuffer* tbuf = mergedmesh->getTransformsBuffer();

    unsigned int numVertices = vbuf->getNumItems() ;
    unsigned int numFaces = ibuf->getNumItems()/3;    
    unsigned int numTransforms = tbuf ? tbuf->getNumItems() : 0  ;    

    geometry->setPrimitiveCount(numFaces);

    LOG(info) << "OGeo::makeGeometry"
              << " numVertices " << numVertices 
              << " numFaces " << numFaces
              << " numTransforms " << numTransforms 
              ;


    optix::Buffer vertexBuffer = createInputBuffer<optix::float3>( mergedmesh->getVerticesBuffer(), RT_FORMAT_FLOAT3 );
    geometry["vertexBuffer"]->setBuffer(vertexBuffer);

    optix::Buffer indexBuffer = createInputBuffer<optix::int3>( mergedmesh->getIndicesBuffer(), RT_FORMAT_INT3, 3 ); 
    geometry["indexBuffer"]->setBuffer(indexBuffer);

    optix::Buffer nodeBuffer = createInputBuffer<unsigned int>( mergedmesh->getNodesBuffer(), RT_FORMAT_UNSIGNED_INT );
    geometry["nodeBuffer"]->setBuffer(nodeBuffer);

    optix::Buffer boundaryBuffer = createInputBuffer<unsigned int>( mergedmesh->getBoundariesBuffer(), RT_FORMAT_UNSIGNED_INT );
    geometry["boundaryBuffer"]->setBuffer(boundaryBuffer);
 
    optix::Buffer sensorBuffer = createInputBuffer<unsigned int>( mergedmesh->getSensorsBuffer(), RT_FORMAT_UNSIGNED_INT );
    geometry["sensorBuffer"]->setBuffer(sensorBuffer);

    optix::Buffer emptyBuffer = m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, 0);
    geometry["tangentBuffer"]->setBuffer(emptyBuffer);
    geometry["bitangentBuffer"]->setBuffer(emptyBuffer);
    geometry["normalBuffer"]->setBuffer(emptyBuffer);
    geometry["texCoordBuffer"]->setBuffer(emptyBuffer);
 
    return geometry ; 
}



template <typename T>
optix::Buffer OGeo::createInputBuffer(GBuffer* buf, RTformat format, unsigned int fold)
{
   unsigned int bytes = buf->getNumBytes() ;
   unsigned int nit = buf->getNumItems()/fold ;
   unsigned int nel = buf->getNumElements();
   unsigned int mul = RayTraceConfig::getMultiplicity(format) ;

   LOG(info)<<"OGeo::createInputBuffer"
            << " bytes " << bytes
            << " nit " << nit 
            << " nel " << nel 
            << " mul " << mul 
            << " fold " << fold 
            << " sizeof(T) " << sizeof(T)
            ;

   assert(sizeof(T)*nit == buf->getNumBytes() );
   assert(nel == mul/fold );

   optix::Buffer buffer = m_context->createBuffer( RT_BUFFER_INPUT, format, nit );
   memcpy( buffer->map(), buf->getPointer(), buf->getNumBytes() );
   buffer->unmap();

   return buffer ; 
}


/*
optix::float3 OGeo::getMin()
{
    return optix::make_float3(0.f, 0.f, 0.f); 
}

optix::float3 OGeo::getMax()
{
    return optix::make_float3(0.f, 0.f, 0.f); 
}
*/


