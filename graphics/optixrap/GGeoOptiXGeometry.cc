#include "GGeoOptiXGeometry.hh"

#include <optixu/optixu_vector_types.h>
#include "cu/enums.h"

#include "RayTraceConfig.hh"
#include "GGeo.hh"
#include "GSolid.hh"
#include "GMesh.hh"
#include "GSubstanceLib.hh"
#include "GSubstance.hh"
#include "GPropertyMap.hh"



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
    convertSubstances();
    convertStructure();
}


void GGeoOptiXGeometry::convertSubstances()
{
    printf("GGeoOptiXGeometry::convertSubstances\n"); 
    GSubstanceLib* lib = m_ggeo->getSubstanceLib();
    unsigned int nsub = lib->getNumSubstances();
    for(unsigned int i=0 ; i < nsub ; i++)
    {
        GSubstance* substance = lib->getSubstance(i);
        optix::Material material = convertSubstance(substance);
        m_materials.push_back(material);
    }
    assert(m_materials.size() == nsub);
    printf("GGeoOptiXGeometry::convertSubstances converted %d substances into optix materials \n", nsub); 
}



void GGeoOptiXGeometry::convertStructure()
{
    m_gis.clear();
    traverseNode( m_ggeo->getSolid(0), 0, true );
    assert(m_gis.size() > 0);
    printf("GGeoOptiXGeometry::convertStructure :  converted %lu gi \n", m_gis.size() );
}


void GGeoOptiXGeometry::traverseNode(GNode* node, unsigned int depth, bool recurse)
{
    GSolid* solid = dynamic_cast<GSolid*>(node) ;

    if(solid->isSelected())
    {
        optix::GeometryInstance gi = convertGeometryInstance(solid);
        m_gis.push_back(gi);
        m_ggeo->updateBounds(solid);
    }

    if(recurse)
    {
        for(unsigned int i = 0; i < node->getNumChildren(); i++) traverseNode(node->getChild(i), depth + 1, recurse);
    }
}


optix::GeometryInstance GGeoOptiXGeometry::convertGeometryInstance(GSolid* solid)
{
    optix::Geometry geometry = convertGeometry(solid) ;  




    std::vector<unsigned int>& substanceIndices = solid->getDistinctSubstanceIndices();
    assert(substanceIndices.size() == 1 );  // for now, maybe >1 for merged meshes 

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




optix::Geometry GGeoOptiXGeometry::convertDrawable(GDrawable* drawable)
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

    //GBuffer* nbuf = drawable->getNormalsBuffer();
    //GBuffer* cbuf = drawable->getColorsBuffer();
    //GBuffer* tbuf = drawable->getTexcoordsBuffer();

    unsigned int numVertices = vbuf->getNumItems() ;
    unsigned int numFaces = ibuf->getNumItems();

    geometry->setPrimitiveCount(numFaces);

    {
        assert(sizeof(optix::float3)*numVertices == vbuf->getNumBytes() && vbuf->getNumElements() == 3);
        optix::Buffer vertexBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, numVertices);
        geometry["vertexBuffer"]->setBuffer(vertexBuffer);
        memcpy( vertexBuffer->map(), vbuf->getPointer(), vbuf->getNumBytes() );
        vertexBuffer->unmap();
    }
    {
        assert(sizeof(optix::int3)*numFaces == ibuf->getNumBytes() && ibuf->getNumElements() == 3); 
        optix::Buffer indexBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT3, numFaces );
        geometry["indexBuffer"]->setBuffer(indexBuffer);
        memcpy( indexBuffer->map(), ibuf->getPointer(), ibuf->getNumBytes() );
        indexBuffer->unmap();
    }

    // hmm tempting to merge some of these buffers
    {
        assert(sizeof(unsigned int)*numFaces == dbuf->getNumBytes() && dbuf->getNumElements() == 1);
        optix::Buffer nodeBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, numFaces );
        geometry["nodeBuffer"]->setBuffer(nodeBuffer);
        memcpy( nodeBuffer->map(), dbuf->getPointer(), dbuf->getNumBytes() );
        nodeBuffer->unmap();
    }
    {
        assert(sizeof(unsigned int)*numFaces == sbuf->getNumBytes() && sbuf->getNumElements() == 1);
        optix::Buffer substanceBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, numFaces );
        geometry["substanceBuffer"]->setBuffer(substanceBuffer);
        memcpy( substanceBuffer->map(), sbuf->getPointer(), sbuf->getNumBytes() );
        substanceBuffer->unmap();
    }


    optix::Buffer emptyBuffer = m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, 0);
    geometry["tangentBuffer"]->setBuffer(emptyBuffer);
    geometry["bitangentBuffer"]->setBuffer(emptyBuffer);
    geometry["normalBuffer"]->setBuffer(emptyBuffer);
    geometry["texCoordBuffer"]->setBuffer(emptyBuffer);
 
    return geometry ; 
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


   // hmm maybe use int4 for indexBuffer and stuff node index into slot4 rather than using separate nodeBuffer 
    optix::Buffer nodeBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, numFaces );
    unsigned int* nodeBuffer_Host = static_cast<unsigned int*>( nodeBuffer->map() );
    geometry["nodeBuffer"]->setBuffer(nodeBuffer);
    memcpy( static_cast<void*>( nodeBuffer_Host ),
            static_cast<void*>( solid->getNodeIndices() ),
            sizeof(unsigned int)*numFaces); 
    nodeBuffer->unmap();


    //
    // huh, is this in use ? NOT CURRENTLY
    // while have one substance per mesh this isnt needed, if move to 
    // merged meshes will need to identify which substance corresponds to which face
    //
    optix::Buffer substanceBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, numFaces );
    unsigned int* substanceBuffer_Host = static_cast<unsigned int*>( substanceBuffer->map() );
    geometry["substanceBuffer"]->setBuffer(substanceBuffer);
    memcpy( static_cast<void*>( substanceBuffer_Host ),
            static_cast<void*>( solid->getSubstanceIndices() ),
            sizeof(unsigned int)*numFaces); 
    substanceBuffer->unmap();

    optix::Buffer emptyBuffer = m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, 0);
    geometry["tangentBuffer"]->setBuffer(emptyBuffer);
    geometry["bitangentBuffer"]->setBuffer(emptyBuffer);
    geometry["normalBuffer"]->setBuffer(emptyBuffer);
    geometry["texCoordBuffer"]->setBuffer(emptyBuffer);
 
    return geometry ; 
}



void GGeoOptiXGeometry::addWavelengthTexture(optix::Material& material, GPropertyMap* ptex)
{
    GDomain<double>* domain = ptex->getStandardDomain();
    material["wavelength_domain"]->setFloat(domain->getLow(), domain->getHigh(), domain->getStep() ); 

    unsigned int nprop = ptex->getNumProperties() ;
    assert(nprop % 4 == 0 );           

    const unsigned int nx = domain->getLength(); 
    const unsigned int ny = nprop/4 ; 

    assert(nx == 39 && ny == 4);
    //printf("GGeoOptiXGeometry::addWavelengthTexture nx %u ny %u \n", nx, ny );

    optix::Buffer wavelengthBuffer = m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, nx, ny );
    float* buffer_data = static_cast<float*>( wavelengthBuffer->map() );
  
    // cf sutil/HDRLoader.cpp 
    for( unsigned int j = 0; j < ny; ++j ) 
    { 
        unsigned int offset = j*ny ;  
        GPropertyD* p0 = ptex->getPropertyByIndex(offset+0) ;
        GPropertyD* p1 = ptex->getPropertyByIndex(offset+1) ;
        GPropertyD* p2 = ptex->getPropertyByIndex(offset+2) ;
        GPropertyD* p3 = ptex->getPropertyByIndex(offset+3) ;

        for( unsigned int i = 0; i < nx; ++i ) 
        { 
            unsigned int buf_index = ( j*nx + i )*4;  
            buffer_data[buf_index+0] = p0->getValue(i) ;
            buffer_data[buf_index+1] = p1->getValue(i) ;
            buffer_data[buf_index+2] = p2->getValue(i) ;
            buffer_data[buf_index+3] = p3->getValue(i) ;
#if 0
            printf("GGeoOptiXGeometry::addWavelengthTexture i %2u j %2u buf_index %4u offset %u buf  %10.3f %10.3f %10.3f %10.3f \n",
               i,j,buf_index,offset,
               buffer_data[buf_index+0],
               buffer_data[buf_index+1],
               buffer_data[buf_index+2],
               buffer_data[buf_index+3]);
#endif
        }    
    }
    wavelengthBuffer->unmap(); 


    optix::TextureSampler sampler = m_context->createTextureSampler();

    sampler->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE ); 
    sampler->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE );
    sampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
    sampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);  
    sampler->setIndexingMode(RT_TEXTURE_INDEX_ARRAY_INDEX);  // by inspection : zero based array index offset by 0.5
    sampler->setMaxAnisotropy(1.0f);  
    sampler->setMipLevelCount(1u);     
    sampler->setArraySize(1u);        
    sampler->setBuffer(0u, 0u, wavelengthBuffer);

    material["wavelength_texture"]->setTextureSampler(sampler);


    // 
    // RT_TEXTURE_INDEX_NORMALIZED_COORDINATES
    //
    // RT_TEXTURE_READ_NORMALIZED_FLOAT : 
    //    texture read results automatically converted to normalized float values 
    //    (no normalization is apparent)
    //
    // RT_TEXTURE_READ_ELEMENT_TYPE
    //     ?
    //
    // OptiX currently supports only a single MIP level and a single element texture array, 
    // so many of above settings are boilerplate.
    // Mipmaps are power of 2 reductions of textures, 
    // so can apply appropriate sized texture at different object distances
    //
    // http://en.wikipedia.org/wiki/Anisotropic_filtering
    // for non-isotropic mipmap selection when viewing textured surfaces at oblique angles 
    // ie using different level of details in different dimensions of the texture 
    //
    // A MaxAnisotropy value greater than 0 will enable anisotropic filtering at the specified value.
    // (assuming this to be irrelevant in current OptiX)
}




optix::Material GGeoOptiXGeometry::convertSubstance(GSubstance* substance)
{
    //
    // Each GSubstance (representing boundaries between materials) 
    // is 1-1 related to optix::Material
    //

    RayTraceConfig* cfg = RayTraceConfig::getInstance();

    optix::Material material = m_context->createMaterial();

    unsigned int raytype_radiance = 0 ;
    material->setClosestHitProgram(raytype_radiance, cfg->createProgram("material1_radiance.cu", "closest_hit_radiance"));

    GSubstanceLib* lib = m_ggeo->getSubstanceLib();
    GPropertyMap* ptex = lib->createStandardProperties("ptex", substance);
    substance->setTexProps(ptex);
    //substance->dumpTexProps("GGeoOptiXGeometry::convertSubstance", 510.f ); 

    addWavelengthTexture(material, ptex);

    unsigned int index = substance->getIndex();
    material["contrast_color"]->setFloat(cfg->make_contrast_color(index));   // just for debugging

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


