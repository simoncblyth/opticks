#include "GMergedMeshOptiXGeometry.hh"
#include "OptiXEngine.hh"
#include "GMergedMesh.hh"
#include "GBoundaryLib.hh"

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
    optix::GeometryInstance gi = convertDrawableInstance(m_mergedmesh);
    m_gis.push_back(gi);
}


optix::TextureSampler GMergedMeshOptiXGeometry::makeWavelengthSampler(GBuffer* buffer)
{
   // handles different numbers of substances, but uses static domain length
    unsigned int domainLength = GBoundaryLib::DOMAIN_LENGTH ;
    unsigned int numElementsTotal = buffer->getNumElementsTotal();
    assert( numElementsTotal % domainLength == 0 );

    unsigned int nx = domainLength ;
    unsigned int ny = numElementsTotal / domainLength ;

    LOG(info) << "GMergedMeshOptiXGeometry::makeWavelengthSampler "
              << " numElementsTotal " << numElementsTotal  
              << " (nx)domainLength " << domainLength 
              << " ny (props*subs)  " << ny 
              << " ny/16 " << ny/16 ; 

    optix::Buffer optixBuffer = m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, nx, ny );
    memcpy( optixBuffer->map(), buffer->getPointer(), buffer->getNumBytes() );
    optixBuffer->unmap(); 

    optix::TextureSampler sampler = m_context->createTextureSampler();
    sampler->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE ); 
    sampler->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE );

    RTfiltermode minification = RT_FILTER_LINEAR ;
    RTfiltermode magnification = RT_FILTER_LINEAR ;
    RTfiltermode mipmapping = RT_FILTER_NONE ;
    sampler->setFilteringModes(minification, magnification, mipmapping);

    sampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);  
    sampler->setIndexingMode(RT_TEXTURE_INDEX_ARRAY_INDEX);  // by inspection : zero based array index offset by 0.5
    sampler->setMaxAnisotropy(1.0f);  
    sampler->setMipLevelCount(1u);     
    sampler->setArraySize(1u);        
    sampler->setBuffer(0u, 0u, optixBuffer);

    return sampler ; 
}




optix::TextureSampler GMergedMeshOptiXGeometry::makeReemissionSampler(GBuffer* buffer)
{
    unsigned int domainLength = buffer->getNumElementsTotal();
    unsigned int nx = domainLength ;
    unsigned int ny = 1 ;

    LOG(info) << "GMergedMeshOptiXGeometry::makeReemissionSampler "
              << " (nx)domainLength " << domainLength 
              << " ny " << ny  ;

    optix::Buffer optixBuffer = m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, nx, ny );
    memcpy( optixBuffer->map(), buffer->getPointer(), buffer->getNumBytes() );
    optixBuffer->unmap(); 

    optix::TextureSampler sampler = m_context->createTextureSampler();
    sampler->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE ); 
    sampler->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE );

    RTfiltermode minification = RT_FILTER_LINEAR ;
    RTfiltermode magnification = RT_FILTER_LINEAR ;
    RTfiltermode mipmapping = RT_FILTER_NONE ;
    sampler->setFilteringModes(minification, magnification, mipmapping);

    sampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);  
    sampler->setIndexingMode(RT_TEXTURE_INDEX_ARRAY_INDEX);  // by inspection : zero based array index offset by 0.5
    sampler->setMaxAnisotropy(1.0f);  
    sampler->setMipLevelCount(1u);     
    sampler->setArraySize(1u);        
    sampler->setBuffer(0u, 0u, optixBuffer);

    return sampler ; 
}







optix::Material GMergedMeshOptiXGeometry::makeMaterial()  
{
    optix::Material material = m_context->createMaterial();

    RayTraceConfig* cfg = RayTraceConfig::getInstance();

    material->setClosestHitProgram(OptiXEngine::e_radiance_ray, cfg->createProgram("material1_radiance.cu", "closest_hit_radiance"));

    material->setClosestHitProgram(OptiXEngine::e_propagate_ray, cfg->createProgram("material1_propagate.cu", "closest_hit_propagate"));

    return material ; 
}

optix::float4 GMergedMeshOptiXGeometry::getDomain()
{
    float domain_range = (GBoundaryLib::DOMAIN_HIGH - GBoundaryLib::DOMAIN_LOW); 
    return optix::make_float4(GBoundaryLib::DOMAIN_LOW, GBoundaryLib::DOMAIN_HIGH, GBoundaryLib::DOMAIN_STEP, domain_range); 
}

optix::float4 GMergedMeshOptiXGeometry::getDomainReciprocal()
{
    // only endpoints used for sampling, not the step 
    return optix::make_float4(1./GBoundaryLib::DOMAIN_LOW, 1./GBoundaryLib::DOMAIN_HIGH, 0.f, 0.f); // not flipping order 
}



optix::GeometryInstance GMergedMeshOptiXGeometry::convertDrawableInstance(GMergedMesh* mergedmesh)
{
    optix::Geometry geometry = convertDrawable(mergedmesh) ;  

    // maybe go for single material, with substanceIndex attribute 

    LOG(info) << "GMergedMeshOptiXGeometry::convertDrawableInstance using single material  " ; 

    GBuffer* wavelengthBuffer = mergedmesh->getWavelengthBuffer();
    optix::TextureSampler wavelengthSampler = makeWavelengthSampler(wavelengthBuffer);

    optix::float4 wavelengthDomain = getDomain();
    optix::float4 wavelengthDomainReciprocal = getDomainReciprocal();

    m_context["wavelength_texture"]->setTextureSampler(wavelengthSampler);
    m_context["wavelength_domain"]->setFloat(wavelengthDomain); 
    m_context["wavelength_domain_reciprocal"]->setFloat(wavelengthDomainReciprocal); 


    GBuffer* reemissionBuffer = mergedmesh->getReemissionBuffer();
    float reemissionStep = 1.f/reemissionBuffer->getNumElementsTotal() ; 
    optix::float4 reemissionDomain = optix::make_float4(0.f , 1.f, reemissionStep, 0.f );
    optix::TextureSampler reemissionSampler = makeReemissionSampler(reemissionBuffer);
    m_context["reemission_texture"]->setTextureSampler(reemissionSampler);
    m_context["reemission_domain"]->setFloat(reemissionDomain);

    optix::Material material = makeMaterial();
    std::vector<optix::Material> materials ;
    materials.push_back(material);
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
    GBuffer* bbuf = drawable->getBoundariesBuffer();

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
        assert(sizeof(unsigned int)*numFaces == bbuf->getNumBytes() && bbuf->getNumElements() == 1);
        optix::Buffer boundaryBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, numFaces );
        memcpy( boundaryBuffer->map(), bbuf->getPointer(), bbuf->getNumBytes() );
        boundaryBuffer->unmap();
        geometry["boundaryBuffer"]->setBuffer(boundaryBuffer);
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



