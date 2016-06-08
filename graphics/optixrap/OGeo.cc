#include "OGeo.hh"
#include "OContext.hh"

#ifdef WITH_OPENGL
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#endif

#include <algorithm>
#include <iomanip>

#include <optix_world.h>

// opticks-
#include "Opticks.hh"

// optixrap-
#include "OContext.hh"
#include "OConfig.hh"

#include "GGeo.hh"
#include "GMergedMesh.hh"
#include "GParts.hh"

// npy-
#include "BLog.hh"
#include "NPY.hpp"
#include "NSlice.hpp"
#include "stringutil.hh"


#include "OConfig.hh"


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
//                           repeated (GeometryGroup)
//                     xform_1
//                           repeated (GeometryGroup)
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
//
//  With instancing and ability to identify the intersected instance
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//   Need to be able to assign an index to each instance ...
//   means need a GeometryInstance beneath the xform ? 
//
//   The transform must be assigned exactly one child of type rtGroup, rtGeometryGroup, rtTransform, or rtSelector,
//
//
//
//
//  TODO:
//     Currently all the accelerations are using Sbvh/Bvh.
//     Investigate if container groups might be better as "pass through" 
//     NoAccel as the geometryGroup and groups they contain have all the 
//     geometry.
//

const char* OGeo::BUILDER = "Sbvh" ; 
const char* OGeo::TRAVERSER = "Bvh" ; 


const char* OGeo::description(const char* msg)
{
    if(!m_description)
    {
        char desc[128];
        snprintf(desc, 128, "%s %s %s ", msg, m_builder, m_traverser );
        m_description = strdup(desc); 
    }
    return m_description ;
}

void OGeo::init()
{
    m_context = m_ocontext->getContext();
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

    LOG(info) << "OGeo::convert" << " nmm " << nmm ;

    for(unsigned int i=0 ; i < nmm ; i++)
    {
        GMergedMesh* mm = m_ggeo->getMergedMesh(i); 
        assert(mm);

        if( mm->isSkip())
        {
            LOG(warning) << "OGeo::convert"
                         << " skipping mesh " << i 
                         ;
            continue ; 
        }

        if( i == 0 )
        {
            optix::Geometry gmm = makeGeometry(mm);
            optix::Material mat = makeMaterial();
            optix::GeometryInstance gi = makeGeometryInstance(gmm,mat);
            gi["instance_index"]->setUint( 0u );  // so same code can run Instanced or not 
            gi["primitive_count"]->setUint( 0u ); // not needed for non-instanced
            m_geometry_group->addChild(gi);
        }
        else
        {
            optix::Group group = makeRepeatedGroup(mm);
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



optix::Group OGeo::makeRepeatedGroup(GMergedMesh* mm)
{
    NPY<float>* itransforms = mm->getITransformsBuffer();

    NSlice* islice = mm->getInstanceSlice(); 
    if(!islice) islice = new NSlice(0, itransforms->getNumItems()) ;

    unsigned int numTransforms = islice->count();
    assert(itransforms && numTransforms > 0);

    NPY<unsigned int>* ibuf = mm->getInstancedIdentityBuffer();
    unsigned int numIdentity = ibuf->getNumItems();

    assert(numIdentity % numTransforms == 0 && "expecting numIdentity to be integer multiple of numTransforms"); 
    unsigned int numSolids = numIdentity/numTransforms ;

    LOG(info) << "OGeo::makeRepeatedGroup"
              << " numTransforms " << numTransforms 
              << " numIdentity " << numIdentity  
              << " numSolids " << numSolids  
              << " islice " << islice->description() 
              ; 


    float* tptr = (float*)itransforms->getPointer(); 

    optix::Group assembly = m_context->createGroup();
    assembly->setChildCount(islice->count());

    optix::Geometry gmm = makeGeometry(mm);
    optix::Material mat = makeMaterial();

    optix::Acceleration accel = makeAcceleration() ;
    // common accel for all instances 

    bool transpose = true ; 

    unsigned int ichild = 0 ; 

    for(unsigned int i=islice->low ; i<islice->high ; i+=islice->step)
    {
        optix::Transform xform = m_context->createTransform();
        assembly->setChild(ichild, xform);

        // proliferating *pergi* so can assign an instance index to it 
        optix::GeometryInstance pergi = makeGeometryInstance(gmm, mat); 
        pergi["instance_index"]->setUint( i );

        optix::GeometryGroup perxform = m_context->createGeometryGroup();
        perxform->addChild(pergi);
        perxform->setAcceleration( accel );

        xform->setChild(perxform);

        const float* tdata = tptr + 16*i ; 
        optix::Matrix4x4 m(tdata) ;
        xform->setMatrix(transpose, m.getData(), 0);

        ichild++ ;
        //dump("OGeo::makeRepeatedGroup", m.getData());
    }
    return assembly ;

/*
   After instance ID possible
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

          assembly        (Group) 
             xform        (Transform)
               perxform   (GeometryGroup)
                 pergi    (GeometryInstance)      
                     gmm  (Geometry)               the same gmm and mat are child of all xform/perxform/pergi
                     mat  (Material) 
             xform        (Transform)
               perxform   (GeometryGroup)
                  pergi   (GeometryInstance)      
                     gmm  (Geometry)
                     mat  (Material) 
             ...
*/
}




/*
optix::Group OGeo::PRIOR_makeRepeatedGroup(GMergedMesh* mm, unsigned int limit)
{
    assert(0);
    NPY<float>* tbuf = mm->getITransformsBuffer();
    unsigned int numTransforms = limit > 0 ? std::min(tbuf->getNumItems(), limit) : tbuf->getNumItems() ;
    assert(tbuf && numTransforms > 0);

    LOG(info) << "OGeo::makeRepeatedGroup numTransforms " << numTransforms ; 

    float* tptr = (float*)tbuf->getPointer(); 

    optix::Group assembly = m_context->createGroup();
    assembly->setChildCount(numTransforms);

    optix::GeometryGroup repeated = m_context->createGeometryGroup();
    optix::Geometry gmm = makeGeometry(mm);
    optix::Material mat = makeMaterial();
    optix::GeometryInstance gi = makeGeometryInstance(gmm, mat); 
    repeated->addChild(gi);
    repeated->setAcceleration( makeAcceleration() );

    bool transpose = true ; 
    for(unsigned int i=0 ; i<numTransforms ; i++)
    {
        optix::Transform xform = m_context->createTransform();
        assembly->setChild(i, xform);
        xform->setChild(repeated);
        const float* tdata = tptr + 16*i ; 
        optix::Matrix4x4 m(tdata) ;
        xform->setMatrix(transpose, m.getData(), 0);
        //dump("OGeo::makeRepeatedGroup", m.getData());
    }
    return assembly ;

}
*/

   /*
   Before instance ID possible
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

          assembly        (Group) 
             xform_0      (Transform)

               repeated   (GeometryGroup)         exact same repeated group is child of all xform
                  gi      (GeometryInstance)     
                     gmm  (Geometry)
                     mat  (Material) 

             xform_1      (Transform)

               repeated   (GeometryGroup)
                  gi      (GeometryInstance)    
                     gmm  (Geometry)
                     mat  (Material) 
             ...

   */




void OGeo::dump(const char* msg, const float* f)
{
    printf("%s\n", msg);
    for(unsigned int i=0 ; i < 16 ; i++) printf(" %10.3f ", *(f+i) ) ;
    printf("\n");
}


optix::Acceleration OGeo::makeAcceleration(const char* builder, const char* traverser)
{
    const char* ubuilder = builder ? builder : m_builder ;
    const char* utraverser = traverser ? traverser : m_traverser ;

    LOG(debug) << "OGeo::makeAcceleration " 
              << " ubuilder " << ubuilder 
              << " utraverser " << utraverser
              ; 
 
    optix::Acceleration acceleration = m_context->createAcceleration(ubuilder, utraverser );
    acceleration->setProperty( "vertex_buffer_name", "vertexBuffer" );
    acceleration->setProperty( "index_buffer_name", "indexBuffer" );
    //acceleration->markDirty();
    return acceleration ; 
}

optix::Material OGeo::makeMaterial()
{
    LOG(info) << "OGeo::makeMaterial " 
               << " radiance_ray " << OContext::e_radiance_ray  
               << " propagate_ray " << OContext::e_propagate_ray  
               ; 

    optix::Material material = m_context->createMaterial();
    material->setClosestHitProgram(OContext::e_radiance_ray, m_ocontext->createProgram("material1_radiance.cu.ptx", "closest_hit_radiance"));
    material->setClosestHitProgram(OContext::e_propagate_ray, m_ocontext->createProgram("material1_propagate.cu.ptx", "closest_hit_propagate"));
    return material ; 
}

optix::GeometryInstance OGeo::makeGeometryInstance(optix::Geometry geometry, optix::Material material)
{
    LOG(debug) << "OGeo::makeGeometryInstance material1  " ; 

    std::vector<optix::Material> materials ;
    materials.push_back(material);
    optix::GeometryInstance gi = m_context->createGeometryInstance( geometry, materials.begin(), materials.end()  );  

    return gi ;
}

optix::Geometry OGeo::makeGeometry(GMergedMesh* mergedmesh)
{
    optix::Geometry geometry ; 
    const char geocode = mergedmesh->getGeoCode();
    switch(geocode)
    { 
        case Opticks::GEOCODE_TRIANGULATED:
                   geometry = makeTriangulatedGeometry(mergedmesh);
                   break ; 
        case Opticks::GEOCODE_ANALYTIC:
                   geometry = makeAnalyticGeometry(mergedmesh);
                   break ; 
        default:
                   LOG(fatal) << "OGeo::makeGeometry geocode must be triangulated or analytic, not [" << (char)geocode  << "]" ;
                   assert(0);
                   break ; 
    }
    return geometry ; 

}


optix::Geometry OGeo::makeAnalyticGeometry(GMergedMesh* mm)
{
    // when using --test eg PmtInBox or BoxInBox the mesh is fabricated in GGeoTest

    GParts* pmt = mm->getParts();
    assert(pmt && "GMergedMesh with GEOCODE_ANALYTIC must have associated GParts, see GGeo::modifyGeometry "); 

    if(pmt->getSolidBuffer() == NULL)
    {
        LOG(debug) << "OGeo::makeAnalyticGeometry closing GParts analytic geometry" ; 
        pmt->close();
        LOG(debug) << "OGeo::makeAnalyticGeometry closing GParts analytic geometry DONE" ; 
    }


    if(m_verbose)
    pmt->Summary("OGeo::makeAnalyticGeometry pmt Summary");

    NPY<float>* partBuf = pmt->getPartBuffer();
    NPY<unsigned int>* solidBuf = pmt->getSolidBuffer(); // not a good name, as connection to CSG Solid is weakening

    assert(partBuf);
    assert(solidBuf);


    if(m_verbose)
    {
        partBuf->dump("OGeo::makeAnalyticGeometry partBuf");
        solidBuf->dump("OGeo::makeAnalyticGeometry solidBuf partOffset/numParts/solidIndex/0");
        //partBuf->save("/tmp/partBuf.npy");
        //solidBuf->save("/tmp/solidBuf.npy");
    }

    NPY<unsigned int>* idBuf = mm->getAnalyticInstancedIdentityBuffer();
    assert(idBuf);

    NPY<float>* itransforms = mm->getITransformsBuffer();
    unsigned int numITransforms = itransforms ? itransforms->getNumItems() : 0  ;    
    assert(idBuf->getNumItems() == numITransforms );

    unsigned int numSolids = solidBuf->getNumItems();
    unsigned int numParts = partBuf->getNumItems();

    assert( numSolids < 10 );  // expecting small number


    LOG(debug)   << "OGeo::makeAnalyticGeometry " 
                 << " mmIndex " << mm->getIndex() 
                 << " numSolids " << numSolids 
                 << " numParts " << numParts
                 << " numITransforms " << numITransforms 
                 ;

    optix::Geometry geometry = m_context->createGeometry();

    geometry->setPrimitiveCount( numSolids );
    geometry["primitive_count"]->setUint( numSolids );  // needed GPU side, for instanced offsets 

    geometry->setIntersectionProgram(m_ocontext->createProgram("hemi-pmt.cu.ptx", "intersect"));
    geometry->setBoundingBoxProgram(m_ocontext->createProgram("hemi-pmt.cu.ptx", "bounds"));

    optix::Buffer solidBuffer = createInputBuffer<optix::uint4, unsigned int>( solidBuf, RT_FORMAT_UNSIGNED_INT4, 1 , "solidBuffer"); 
    geometry["solidBuffer"]->setBuffer(solidBuffer);

    optix::Buffer partBuffer = createInputBuffer<optix::float4, float>( partBuf, RT_FORMAT_FLOAT4, 1 , "partBuffer"); 
    geometry["partBuffer"]->setBuffer(partBuffer);

    optix::Buffer identityBuffer = createInputBuffer<optix::uint4, unsigned int>( idBuf, RT_FORMAT_UNSIGNED_INT4, 1 , "identityBuffer"); 
    geometry["identityBuffer"]->setBuffer(identityBuffer);


    optix::Buffer prismBuffer = m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT);
    prismBuffer->setFormat(RT_FORMAT_FLOAT4);
    prismBuffer->setSize(5);
    geometry["prismBuffer"]->setBuffer(prismBuffer);


    return geometry ; 
}



optix::Geometry OGeo::makeTriangulatedGeometry(GMergedMesh* mm)
{
    // index buffer items are the indices of every triangle vertex, so divide by 3 to get faces 
    // and use folding by 3 in createInputBuffer
    //
    // DYB: for instanced geometry this just sees the 5 solids of the repeated instance 
    //      and numFaces is the sum of the face counts of those, and numITransforms is 672
    // 
    //  in order to provide identity to the instances need to repeat the iidentity to the
    //  triangles
    //

    optix::Geometry geometry = m_context->createGeometry();
    geometry->setIntersectionProgram(m_ocontext->createProgram("TriangleMesh.cu.ptx", "mesh_intersect"));
    geometry->setBoundingBoxProgram(m_ocontext->createProgram("TriangleMesh.cu.ptx", "mesh_bounds"));

    unsigned int numSolids = mm->getNumSolids();
    unsigned int numFaces = mm->getNumFaces();
    unsigned int numITransforms = mm->getNumITransforms();

    geometry->setPrimitiveCount(numFaces);
    assert(geometry->getPrimitiveCount() == numFaces);
    geometry["primitive_count"]->setUint( geometry->getPrimitiveCount() );  // needed for instanced offsets 

    LOG(info) << "OGeo::makeTriangulatedGeometry " 
              << " mmIndex " << mm->getIndex() 
              << " numFaces (PrimitiveCount) " << numFaces
              << " numSolids " << numSolids
              << " numITransforms " << numITransforms 
              ;


    GBuffer* id = NULL ; 
    if(numITransforms > 0)
    {
        id = mm->getFaceRepeatedInstancedIdentityBuffer();
        assert(id);
        LOG(info) << "OGeo::makeTriangulatedGeometry using FaceRepeatedInstancedIdentityBuffer"
                  << " friid items " << id->getNumItems() 
                  << " numITransforms*numFaces " << numITransforms*numFaces
                  ;

        assert( id->getNumItems() == numITransforms*numFaces );
   }
   else
   {
        id = mm->getFaceRepeatedIdentityBuffer();
        assert(id);
        LOG(info) << "OGeo::makeTriangulatedGeometry using FaceRepeatedIdentityBuffer"
                  << " frid items " << id->getNumItems() 
                  << " numFaces " << numFaces
                  ;
        assert( id->getNumItems() == numFaces );
   }  

   optix::Buffer identityBuffer = createInputBuffer<optix::uint4>( id, RT_FORMAT_UNSIGNED_INT4, 1 , "identityBuffer"); 
   geometry["identityBuffer"]->setBuffer(identityBuffer);

 
    // TODO: purloin the OpenGL buffers to avoid duplicating the geometry info on GPU 
    // TODO : attempt to isolate this bad behavior 
    // TODO : float3 is a known to be problematic, maybe try with float4
    // TODO : consolidate the three RT_FORMAT_UNSIGNED_INT into one UINT4
    // setting reuse to true causes OptiX launch failure : bad enum 
    bool reuse = false ;  

    optix::Buffer vertexBuffer = createInputBuffer<optix::float3>( mm->getVerticesBuffer(), RT_FORMAT_FLOAT3, 1, "vertexBuffer", reuse ); 
    geometry["vertexBuffer"]->setBuffer(vertexBuffer);

    optix::Buffer indexBuffer = createInputBuffer<optix::int3>( mm->getIndicesBuffer(), RT_FORMAT_INT3, 3 , "indexBuffer");  // need the 3 to fold for faces
    geometry["indexBuffer"]->setBuffer(indexBuffer);


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
   //TODO: eliminate use of this, moving to NPY buffers instead

   unsigned int bytes = buf->getNumBytes() ;

   unsigned int bit = buf->getNumItems() ; 
   unsigned int nit = bit/fold ; 
   unsigned int nel = buf->getNumElements();
   unsigned int mul = OConfig::getMultiplicity(format) ;

   int buffer_id = buf->getBufferId() ;


   if(m_verbose)
   LOG(info)<<"OGeo::createInputBuffer [GBuffer]"
            << " fmt " << std::setw(20) << OConfig::getFormatName(format)
            << " name " << std::setw(20) << name
            << " bytes " << std::setw(8) << bytes
            << " bit " << std::setw(7) << bit 
            << " nit " << std::setw(7) << nit 
            << " nel " << std::setw(3) << nel 
            << " mul " << std::setw(3) << mul 
            << " fold " << std::setw(3) << fold 
            << " sizeof(T) " << std::setw(3) << sizeof(T)
            << " sizeof(T)*nit " << std::setw(3) << sizeof(T)*nit
            << " id " << std::setw(3) << buffer_id 
            ;

   assert(sizeof(T)*nit == buf->getNumBytes() );
   assert(nel == mul/fold );

   optix::Buffer buffer ;

   if(buffer_id > -1 && reuse)
   {
#ifdef WITH_OPENGL
       //Reuse attempt fails, with:  Caught exception: GL error: Invalid enum
        int buffer_target = buf->getBufferTarget();
        glBindBuffer(buffer_target, buffer_id) ;

        buffer = m_context->createBufferFromGLBO(RT_BUFFER_INPUT, buffer_id);
        buffer->setFormat(format); 
        buffer->setSize(nit);

        glBindBuffer(buffer_target, 0) ;
#else
        assert(0); // not compiled WITH_OPENGL 
#endif
   } 
   else
   {
        buffer = m_context->createBuffer( RT_BUFFER_INPUT, format, nit );
        memcpy( buffer->map(), buf->getPointer(), buf->getNumBytes() );
        buffer->unmap();
   } 

   return buffer ; 
}


template <typename T, typename S>
optix::Buffer OGeo::createInputBuffer(NPY<S>* buf, RTformat format, unsigned int fold, const char* name, bool reuse)
{
   unsigned int bytes = buf->getNumBytes() ;
   unsigned int nel = buf->getNumElements();    // size of the last dimension
   unsigned int bit = buf->getNumItems(0,-1) ;  // size of all but the last dimension
   unsigned int nit = bit/fold ; 
   unsigned int mul = OConfig::getMultiplicity(format) ; // eg multiplicity of FLOAT4 is 4

   int buffer_id = buf->getBufferId() ;

   bool from_gl = buffer_id > -1 && reuse ;

   if(m_verbose)
   LOG(info)<<"OGeo::createInputBuffer [NPY<T>] "
            << " fmt " << std::setw(20) << OConfig::getFormatName(format)
            << " name " << std::setw(20) << name
            << " bytes " << std::setw(8) << bytes
            << " bit " << std::setw(7) << bit 
            << " nit " << std::setw(7) << nit 
            << " nel " << std::setw(3) << nel 
            << " mul " << std::setw(3) << mul 
            << " fold " << std::setw(3) << fold 
            << " sizeof(T) " << std::setw(3) << sizeof(T)
            << " sizeof(T)*nit " << std::setw(3) << sizeof(T)*nit
            << " id " << std::setw(3) << buffer_id 
            << " from_gl " << from_gl 
            ;


   // typical T is optix::float4, typically NPY buffers should arrange last dimension 4 
   assert(sizeof(T)*nit == buf->getNumBytes() );
   assert(nel == mul/fold );

   optix::Buffer buffer ;

   if(from_gl)
   {
#ifdef WITH_OPENGL
       // Reuse attempt fails, getting: Caught exception: GL error: Invalid enum
        int buffer_target = buf->getBufferTarget();
        glBindBuffer(buffer_target, buffer_id) ;

        buffer = m_context->createBufferFromGLBO(RT_BUFFER_INPUT, buffer_id);
        buffer->setFormat(format); 
        buffer->setSize(nit);

        glBindBuffer(buffer_target, 0) ;
#else
        assert(0); // not compiled WITH_OPENGL 
#endif


   } 
   else
   {
        buffer = m_context->createBuffer( RT_BUFFER_INPUT, format, nit );
        memcpy( buffer->map(), buf->getPointer(), buf->getNumBytes() );
        buffer->unmap();
   } 

   return buffer ; 
}


