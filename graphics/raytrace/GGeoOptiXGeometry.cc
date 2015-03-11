#include "GGeoOptiXGeometry.hh"

#include <optixu/optixu_vector_types.h>

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




optix::TextureSampler GGeoOptiXGeometry::makeWavelengthTexture(GSubstance* substance)
{
    optix::TextureSampler sampler = m_context->createTextureSampler();

    GPropertyMap* imat = substance->getInnerMaterial();
    GProperty<float>* abslength = imat->getProperty("ABSLENGTH");
    substance->Summary( abslength ? "imat with ABSLENGTH" :  "imat miss ABSLENGTH" ); 

    //
    // GSubstance incorporates a variety of properties from 
    //
    //    innerMaterial
    //    outerMaterial
    //    innerSurface
    //    outerSurface 
    //
    // * surface props are often not present
    // * need to define a standard set of properties (enum and vector of keys)  
    // * handle missing qtys with default values ?  
    // * need regularized wavelength domain, 
    //   so all props are interpolated to standard wavelengths (like Chroma) 
    // * need to encode the wavelength ranges, so can correctly convert a wavelength into
    //   a texture coordinate to lookup
    // * textures have better performance when there is "spatial locality" ie repeated
    //   lookups tending to be in same region of the texture : so design the property enum
    //   to put properties that are needed together at close tex coordinates  
    //


    const unsigned int nx = 64 ;  // standard number of wavelength samples
    const unsigned int ny = 64 ;  // number of wavelength dependent properties to include in the texture 

    optix::Buffer wavelengthBuffer = m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, nx, ny );
    float* buffer_data = static_cast<float*>( wavelengthBuffer->map() );
  
    // sutil/HDRLoader.cpp  : is this the right buffer layout ?
    for( unsigned int i = 0; i < nx; ++i ) { 
    for( unsigned int j = 0; j < ny; ++j ) { 
        unsigned int buf_index = ( (j)*nx + i )*4;  
        buffer_data[buf_index] = 0.0f ;
        buffer_data[buf_index] = 0.0f ;
        buffer_data[buf_index] = 0.0f ;
        buffer_data[buf_index] = 0.0f ;
    }    
    }
    wavelengthBuffer->unmap(); 

    sampler->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE ); // handling out of range tex coordinates (will that happen, reemission wavelength constrained to range: so no?)
    sampler->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE );

    sampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);

    //sampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES); 
    sampler->setIndexingMode(RT_TEXTURE_INDEX_ARRAY_INDEX);

    sampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT); //  texture read results automatically converted to normalized float values 
    //sampler->setReadMode(RT_TEXTURE_READ_ELEMENT_TYPE);   //  


    // OptiX currently supports only a single MIP level and a single element texture array, 
    // so the below are almost entirely boilerplate
    //
    // mipmaps are power of 2 reductions of textures, 
    // so can apply appropriate sized texture at different object distances
    //
    // http://en.wikipedia.org/wiki/Anisotropic_filtering
    // for non-isotropic mipmap selection when viewing textured surfaces at oblique angles 
    // ie using different level of details in different dimensions of the texture 
    //
    // A MaxAnisotropy value greater than 0 will enable anisotropic filtering at the specified value.
    // (assuming this to be irrelevant in current OptiX)
    //
    sampler->setMaxAnisotropy(1.0f);  
    sampler->setMipLevelCount(1u);     
    sampler->setArraySize(1u);        
    sampler->setBuffer(0u, 0u, wavelengthBuffer);

    return sampler ; 
}



optix::Material GGeoOptiXGeometry::convertSubstance(GSubstance* substance)
{
    RayTraceConfig* cfg = RayTraceConfig::getInstance();

    optix::Material material = m_context->createMaterial();

    unsigned int raytype_radiance = 0 ;
    material->setClosestHitProgram(raytype_radiance, cfg->createProgram("material1.cu", "closest_hit_radiance"));

    unsigned int raytype_touch = 2 ;
    material->setClosestHitProgram(raytype_touch   , cfg->createProgram("material1.cu", "closest_hit_touch"));

    unsigned int index = substance->getIndex();

    material["contrast_color"]->setFloat(cfg->make_contrast_color(index));

    optix::TextureSampler sampler = makeWavelengthTexture(substance);

    material["wavelength_texture"]->setTextureSampler(sampler);

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




