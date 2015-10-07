#include "OGeo.hh"

#include <GL/glew.h>
#include <GLFW/glfw3.h>


#include <algorithm>
#include <iomanip>

#include <optix_world.h>

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


//  Prior to instancing 
//  ~~~~~~~~~~~~~~~~~~~~~~~~
//
//  Simplest possible node tree
//
//         geometry_group 
//             acceleration  
//             geometry_instance 0
//             geometry_instance 1
//             ...            
//
//         1 to 1 mapping of GMergedMesh to "geometry_instance" 
//         each of which comprises one "geometry" with a single "material"
//         which refers to boundary lib properties lodged in the context by OBoundaryLib
//
//  Preparing for instancing
//  ~~~~~~~~~~~~~~~~~~~~~~~~~
//
//   Transforms can only be contained in "group" so 
//   add top level group with another acceleration structure
//
//        group (top)
//           acceleration
//           geometry_group
//                acceleration
//                geometry_instance 0
//                geometry_instance 1
//                 
//
// With instancing
// ~~~~~~~~~~~~~~~~~
//
//         m_top (Group)
//             acceleration
//
//             m_geometry_group (GeometryGroup)
//                 acceleration
//                 geometry_instance 0
//                 geometry_instance 1
//
//             m_repeated_group (Group)
//                 acceleration 
//
//                 group 0
//                     acceleration
//                     xform_0
//                           repeated
//                     xform_1
//                           repeated
//                     ...
// 
//                 group 1
//                      acceleration
//                      xform_0
//                           repeated
//                      ...
//
//
//                  where repeated contains single gi (GeometryInstance) 
//
//  TODO:
//     Currently all the accelerations are using Sbvh/Bvh.
//     Investigate if container groups might be better as "pass through" 
//     NoAccel as the geometryGroup and groups they contain have all the 
//     geometry.
//

const char* OGeo::BUILDER = "Sbvh" ; 
const char* OGeo::TRAVERSER = "Bvh" ; 


void OGeo::init()
{
    m_geometry_group = m_context->createGeometryGroup();
    m_repeated_group = m_context->createGroup();
}

void OGeo::setTop(optix::Group top)
{
    m_top = top ; 
}

void OGeo::convert()
{
    unsigned int nmm = m_ggeo->getNumMergedMesh();
    unsigned int repeatLimit = 0 ;    // for debugging only, 0:no limit 

    LOG(info) << "OGeo::convert"
              << " nmm " << nmm
              << " repeatLimit " << repeatLimit 
              ;

    for(unsigned int i=0 ; i < nmm ; i++)
    {
        GMergedMesh* mm = m_ggeo->getMergedMesh(i); 
        assert(mm);
        if( i == 0 )
        {
            optix::GeometryInstance gi = makeGeometryInstance(mm);
            m_geometry_group->addChild(gi);
        }
        else
        {
            optix::Group group = makeRepeatedGroup(mm, repeatLimit);
            group->setAcceleration( makeAcceleration() );
            m_repeated_group->addChild(group); 
        }
    }

    // all group and geometry_group need to have distinct acceleration structures

    unsigned int geometryGroupCount = m_geometry_group->getChildCount() ;
    unsigned int repeatedGroupCount = m_repeated_group->getChildCount() ;
   
    LOG(info) << "OGeo::convert"
              << " geometryGroupCount " << geometryGroupCount
              << " repeatedGroupCount " << repeatedGroupCount
              ;


    if(geometryGroupCount > 0)
    {
         m_top->addChild(m_geometry_group);
         m_geometry_group->setAcceleration( makeAcceleration() );
    } 

    if(repeatedGroupCount > 0)
    {
         m_top->addChild(m_repeated_group);
         m_repeated_group->setAcceleration( makeAcceleration() );
    } 

    m_top->setAcceleration( makeAcceleration() );
}


optix::Group OGeo::makeRepeatedGroup(GMergedMesh* mm, unsigned int limit)
{
    GBuffer* tbuf = mm->getITransformsBuffer();
    unsigned int numTransforms = limit > 0 ? std::min(tbuf->getNumItems(), limit) : tbuf->getNumItems() ;
    assert(tbuf && numTransforms > 0);

    LOG(info) << "OGeo::makeRepeatedGroup numTransforms " << numTransforms ; 

    float* tptr = (float*)tbuf->getPointer(); 

    optix::Group group = m_context->createGroup();
    group->setChildCount(numTransforms);

    optix::GeometryInstance gi = makeGeometryInstance(mm); 
    optix::GeometryGroup repeated = m_context->createGeometryGroup();
    repeated->addChild(gi);
    repeated->setAcceleration( makeAcceleration() );

    bool transpose = true ; 
    for(unsigned int i=0 ; i<numTransforms ; i++)
    {
        optix::Transform xform = m_context->createTransform();
        group->setChild(i, xform);
        xform->setChild(repeated);
        const float* tdata = tptr + 16*i ; 
        optix::Matrix4x4 m(tdata) ;
        xform->setMatrix(transpose, m.getData(), 0);
        //dump("OGeo::makeRepeatedGroup", m.getData());
    }
    return group ;
}


void OGeo::dump(const char* msg, const float* f)
{
    printf("%s\n", msg);
    for(unsigned int i=0 ; i < 16 ; i++) printf(" %10.3f ", *(f+i) ) ;
    printf("\n");
}


optix::Acceleration OGeo::makeAcceleration(const char* builder, const char* traverser)
{
    const char* ubuilder = builder ? builder : BUILDER ;
    const char* utraverser = traverser ? traverser : TRAVERSER ;

    LOG(info) << "OGeo::makeAcceleration " 
              << " ubuilder " << ubuilder 
              << " utraverser " << utraverser
              ; 
 
    optix::Acceleration acceleration = m_context->createAcceleration(ubuilder, utraverser );
    acceleration->setProperty( "vertex_buffer_name", "vertexBuffer" );
    acceleration->setProperty( "index_buffer_name", "indexBuffer" );
    acceleration->markDirty();
    return acceleration ; 
}


optix::GeometryInstance OGeo::makeGeometryInstance(GMergedMesh* mergedmesh)
{
    LOG(debug) << "OGeo::makeGeometryInstance material1  " ; 

    optix::Material material = m_context->createMaterial();
    RayTraceConfig* cfg = RayTraceConfig::getInstance();
    material->setClosestHitProgram(OEngine::e_radiance_ray, cfg->createProgram("material1_radiance.cu.ptx", "closest_hit_radiance"));
    material->setClosestHitProgram(OEngine::e_propagate_ray, cfg->createProgram("material1_propagate.cu.ptx", "closest_hit_propagate"));

    std::vector<optix::Material> materials ;
    materials.push_back(material);

    optix::Geometry geometry = makeGeometry(mergedmesh) ;  
    optix::GeometryInstance gi = m_context->createGeometryInstance( geometry, materials.begin(), materials.end()  );  

    return gi ;
}

optix::Geometry OGeo::makeGeometry(GMergedMesh* mergedmesh)
{
    optix::Geometry geometry ; 
    switch(mergedmesh->getGeoCode())
    { 
        case 'T':
                   geometry = makeTriangulatedGeometry(mergedmesh);
                   break ; 
        case 'S':
                   geometry = makeSimplifiedGeometry(mergedmesh);
                   break ; 
        default:
                   assert(0);
                   break ; 
    }
    return geometry ; 

}

optix::Geometry OGeo::makeSimplifiedGeometry(GMergedMesh* mergedmesh)
{
    LOG(warning) << "OGeo::makeSimplifiedGeometry " ; 
    return makeTriangulatedGeometry(mergedmesh);

    // replacing instance1 with a sphere positioned to match the cathode front face
    // no triangles...  need a sphere buffer  
}

optix::Geometry OGeo::makeTriangulatedGeometry(GMergedMesh* mergedmesh)
{
    LOG(info) << "OGeo::makeTriangulatedGeometry " ; 
    // index buffer items are the indices of every triangle vertex, so divide by 3 to get faces 
    // and use folding by 3 in createInputBuffer

    optix::Geometry geometry = m_context->createGeometry();
    RayTraceConfig* cfg = RayTraceConfig::getInstance();
    geometry->setIntersectionProgram(cfg->createProgram("TriangleMesh.cu.ptx", "mesh_intersect"));
    geometry->setBoundingBoxProgram(cfg->createProgram("TriangleMesh.cu.ptx", "mesh_bounds"));

    GBuffer* vbuf = mergedmesh->getVerticesBuffer();
    GBuffer* ibuf = mergedmesh->getIndicesBuffer();
    GBuffer* tbuf = mergedmesh->getITransformsBuffer();

    unsigned int numVertices = vbuf->getNumItems() ;
    unsigned int numFaces = ibuf->getNumItems()/3;    
    unsigned int numTransforms = tbuf ? tbuf->getNumItems() : 0  ;    

    geometry->setPrimitiveCount(numFaces);

    LOG(debug) << "OGeo::makeGeometry"
              << " numVertices " << numVertices 
              << " numFaces " << numFaces
              << " numTransforms " << numTransforms 
              ;

    // TODO: purloin the OpenGL buffers to avoid duplicating the geometry info on GPU 
    // TODO : attempt to isolate this bad behavior 
    // TODO : float3 is a known to be problematic, maybe try with float4
    // TODO : consolidate the three RT_FORMAT_UNSIGNED_INT into one UINT4
    //    
    // setting reuse to true causes OptiX launch failure : bad enum 
    bool reuse = false ;  

    optix::Buffer vertexBuffer = createInputBuffer<optix::float3>( mergedmesh->getVerticesBuffer(), RT_FORMAT_FLOAT3, 1, "vertexBuffer", reuse ); 
    geometry["vertexBuffer"]->setBuffer(vertexBuffer);

    optix::Buffer indexBuffer = createInputBuffer<optix::int3>( mergedmesh->getIndicesBuffer(), RT_FORMAT_INT3, 3 , "indexBuffer"); 
    geometry["indexBuffer"]->setBuffer(indexBuffer);

    optix::Buffer nodeBuffer = createInputBuffer<unsigned int>( mergedmesh->getNodesBuffer(), RT_FORMAT_UNSIGNED_INT, 1, "nodeBuffer");
    geometry["nodeBuffer"]->setBuffer(nodeBuffer);

    optix::Buffer boundaryBuffer = createInputBuffer<unsigned int>( mergedmesh->getBoundariesBuffer(), RT_FORMAT_UNSIGNED_INT, 1, "boundaryBuffer");
    geometry["boundaryBuffer"]->setBuffer(boundaryBuffer);
 
    optix::Buffer sensorBuffer = createInputBuffer<unsigned int>( mergedmesh->getSensorsBuffer(), RT_FORMAT_UNSIGNED_INT, 1, "sensorBuffer");
    geometry["sensorBuffer"]->setBuffer(sensorBuffer);

    optix::Buffer emptyBuffer = m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, 0);
    geometry["tangentBuffer"]->setBuffer(emptyBuffer);
    geometry["bitangentBuffer"]->setBuffer(emptyBuffer);
    geometry["normalBuffer"]->setBuffer(emptyBuffer);
    geometry["texCoordBuffer"]->setBuffer(emptyBuffer);
 
    return geometry ; 
}



template <typename T>
optix::Buffer OGeo::createInputBuffer(GBuffer* buf, RTformat format, unsigned int fold, const char* name, bool reuse)
{
   unsigned int bytes = buf->getNumBytes() ;
   unsigned int nit = buf->getNumItems()/fold ;
   unsigned int nel = buf->getNumElements();
   unsigned int mul = RayTraceConfig::getMultiplicity(format) ;

   int buffer_target = buf->getBufferTarget();
   int buffer_id = buf->getBufferId() ;

   LOG(debug)<<"OGeo::createInputBuffer"
            << " fmt " << std::setw(20) << RayTraceConfig::getFormatName(format)
            << " name " << std::setw(20) << name
            << " bytes " << std::setw(8) << bytes
            << " nit " << std::setw(7) << nit 
            << " nel " << std::setw(3) << nel 
            << " mul " << std::setw(3) << mul 
            << " fold " << std::setw(3) << fold 
            << " sizeof(T) " << std::setw(3) << sizeof(T)
            << " id " << std::setw(3) << buffer_id 
            ;

   assert(sizeof(T)*nit == buf->getNumBytes() );
   assert(nel == mul/fold );

   optix::Buffer buffer ;

   //buffer_id = -1 ; // kill attempt to reuse OpenGL buffers

   if(buffer_id > -1 && reuse)
   {
       /*
       Reuse attempt fails, getting 
       Caught exception: GL error: Invalid enum
       */

        glBindBuffer(buffer_target, buffer_id) ;

        buffer = m_context->createBufferFromGLBO(RT_BUFFER_INPUT, buffer_id);
        buffer->setFormat(format); 
        buffer->setSize(nit);

        glBindBuffer(buffer_target, 0) ;
   } 
   else
   {
        buffer = m_context->createBuffer( RT_BUFFER_INPUT, format, nit );
        memcpy( buffer->map(), buf->getPointer(), buf->getNumBytes() );
        buffer->unmap();
   } 

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


