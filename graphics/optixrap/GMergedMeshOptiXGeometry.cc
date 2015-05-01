#include "GMergedMeshOptiXGeometry.hh"
#include "GMergedMesh.hh"

#include "RayTraceConfig.hh"

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal



GMergedMeshOptiXGeometry::GMergedMeshOptiXGeometry(GMergedMesh* mergedmesh)
           : 
           OptiXGeometry(),
           m_mergedmesh(mergedmesh)
{
}


void GMergedMeshOptiXGeometry::convert()
{
    //convertSubstances();

    optix::GeometryInstance gi = convertDrawableInstance(m_mergedmesh);
    m_gis.push_back(gi);
}



optix::GeometryInstance GMergedMeshOptiXGeometry::convertDrawableInstance(GMergedMesh* mergedmesh)
{
    optix::Geometry geometry = convertDrawable(mergedmesh) ;  

    // maybe go for single material, with substanceIndex attribute 

    std::vector<unsigned int>& substanceIndices = mergedmesh->getDistinctSubstances();
    LOG(info) << "GMergedMeshOptiXGeometry::convertDrawableInstance distinct substance indices " << substanceIndices.size() ; 
    std::vector<optix::Material> materials ;
    for(unsigned int i=0 ; i < substanceIndices.size() ; i++)
    {
        unsigned int index = substanceIndices[i];
        optix::Material material = getMaterial(index) ;
        materials.push_back(material); 
    }
    optix::GeometryInstance gi = m_context->createGeometryInstance( geometry, materials.begin(), materials.end()  );  
    return gi ;
}


optix::Geometry GMergedMeshOptiXGeometry::convertDrawable(GMergedMesh* drawable)
{
    // aiming to replace convertGeometry and usage with GMergedMesh
    // where the vertices etc.. have been transformed and combined already  

    optix::Geometry geometry = m_context->createGeometry();

    RayTraceConfig* cfg = RayTraceConfig::getInstance();
    geometry->setIntersectionProgram(cfg->createProgram("TriangleMesh.cu", "mesh_intersect"));
    geometry->setBoundingBoxProgram(cfg->createProgram("TriangleMesh.cu", "mesh_bounds"));

    // contrast with oglrap-/Renderer::gl_upload_buffers
    GBuffer* vbuf = drawable->getVerticesBuffer();
    GBuffer* ibuf = drawable->getIndicesBuffer();
    GBuffer* dbuf = drawable->getNodesBuffer();
    GBuffer* sbuf = drawable->getSubstancesBuffer();

    unsigned int numVertices = vbuf->getNumItems() ;
    unsigned int numFaces = ibuf->getNumItems()/3;    
    // items are the indices so divide by 3 to get faces

    LOG(info) << "GMergedMeshOptiXGeometry::convertDrawable numVertices " << numVertices << " numFaces " << numFaces ;

    geometry->setPrimitiveCount(numFaces);
    {
        assert(sizeof(optix::float3)*numVertices == vbuf->getNumBytes() && vbuf->getNumElements() == 3);
        optix::Buffer vertexBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, numVertices);
        memcpy( vertexBuffer->map(), vbuf->getPointer(), vbuf->getNumBytes() );
        vertexBuffer->unmap();
        geometry["vertexBuffer"]->setBuffer(vertexBuffer);
    }
    {
        assert(sizeof(optix::int3) == 4*3); 
        assert(sizeof(optix::int3)*numFaces == ibuf->getNumBytes()); 
        assert(ibuf->getNumElements() == 1);    // ibuf elements are individual integers
 
        optix::Buffer indexBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT3, numFaces );
        memcpy( indexBuffer->map(), ibuf->getPointer(), ibuf->getNumBytes() );
        indexBuffer->unmap();
        geometry["indexBuffer"]->setBuffer(indexBuffer);
    }
    {
        assert(sizeof(unsigned int)*numFaces == dbuf->getNumBytes() && dbuf->getNumElements() == 1);
        optix::Buffer nodeBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, numFaces );
        memcpy( nodeBuffer->map(), dbuf->getPointer(), dbuf->getNumBytes() );
        nodeBuffer->unmap();
        geometry["nodeBuffer"]->setBuffer(nodeBuffer);
    }
    {
        assert(sizeof(unsigned int)*numFaces == sbuf->getNumBytes() && sbuf->getNumElements() == 1);
        optix::Buffer substanceBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, numFaces );
        memcpy( substanceBuffer->map(), sbuf->getPointer(), sbuf->getNumBytes() );
        substanceBuffer->unmap();
        geometry["substanceBuffer"]->setBuffer(substanceBuffer);
    }


    optix::Buffer emptyBuffer = m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, 0);
    geometry["tangentBuffer"]->setBuffer(emptyBuffer);
    geometry["bitangentBuffer"]->setBuffer(emptyBuffer);
    geometry["normalBuffer"]->setBuffer(emptyBuffer);
    geometry["texCoordBuffer"]->setBuffer(emptyBuffer);
 
    return geometry ; 
}




optix::float3 GMergedMeshOptiXGeometry::getMin()
{
    return optix::make_float3(0.f, 0.f, 0.f); 
}

optix::float3 GMergedMeshOptiXGeometry::getMax()
{
    return optix::make_float3(0.f, 0.f, 0.f); 
}



