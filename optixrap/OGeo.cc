
#include "OGeo.hh"
#include "OGeoStat.hh"
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
#include "OpticksConst.hh"

// optixrap-
#include "OContext.hh"
#include "OConfig.hh"

//#include "GGeo.hh"
//#include "GGeoBase.hh"
#include "GGeoLib.hh"
#include "GMergedMesh.hh"
#include "GParts.hh"

// npy-
#include "PLOG.hh"
#include "NPY.hpp"
#include "NSlice.hpp"
#include "BStr.hh"


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




OGeo::OGeo(OContext* ocontext, Opticks* ok, GGeoLib* geolib, const char* builder, const char* traverser)
           : 
           m_ocontext(ocontext),
           m_ok(ok),
           m_geolib(geolib),
           m_builder(builder ? strdup(builder) : BUILDER),
           m_traverser(traverser ? strdup(traverser) : TRAVERSER),
           m_description(NULL),
           m_verbosity(m_ok->getVerbosity())
{
    init();
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

void OGeo::convert()
{
    unsigned int nmm = m_geolib->getNumMergedMesh();

    if(m_verbosity > 0)
    LOG(info) << "OGeo::convert START  numMergedMesh: " << nmm ;

    for(unsigned i=0 ; i < nmm ; i++)
    {
        convertMergedMesh(i);
    }
    // all group and geometry_group need to have distinct acceleration structures
    unsigned int geometryGroupCount = m_geometry_group->getChildCount() ;
    unsigned int repeatedGroupCount = m_repeated_group->getChildCount() ;
   
    LOG(trace) << "OGeo::convert"
              << " geometryGroupCount(global) " << geometryGroupCount
              << " repeatedGroupCount(instanced) " << repeatedGroupCount
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


    

    if(m_verbosity > 0)
    {
        LOG(info) << "OGeo::convert DONE  numMergedMesh: " << nmm ;
        dumpStats();
    }
}


void OGeo::convertMergedMesh(unsigned i)
{
    if(m_verbosity > 2)
    LOG(info) << "OGeo::convertMesh START " << i ; 

    GMergedMesh* mm = m_geolib->getMergedMesh(i); 

    if( mm == NULL || mm->isSkip() || mm->isEmpty() )
    {
        LOG(warning) << "OGeo::convertMesh"
                     << " skipping mesh " << i 
                     ;
        return  ; 
    }

    // 1st merged mesh is the global non-instanced one
    // subsequent merged meshes contain repeated PMT geometry
    // that typically has analytic primitive intersection implementations  

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


    if(m_verbosity > 2)
    LOG(info) << "OGeo::convertMesh DONE " << i ; 
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

    LOG(trace) << "OGeo::makeRepeatedGroup"
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
    LOG(trace) << "OGeo::makeMaterial " 
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
    if(geocode == OpticksConst::GEOCODE_TRIANGULATED)
    {
        geometry = makeTriangulatedGeometry(mergedmesh);
    }
    else if(geocode == OpticksConst::GEOCODE_ANALYTIC)
    {
        geometry = makeAnalyticGeometry(mergedmesh);
    }
    else
    {
        LOG(fatal) << "OGeo::makeGeometry geocode must be triangulated or analytic, not [" << (char)geocode  << "]" ;
        assert(0);
    }
    return geometry ; 
}



optix::Geometry OGeo::makeAnalyticGeometry(GMergedMesh* mm)
{
    //if(m_verbosity > 2)
    LOG(warning) << "OGeo::makeAnalyticGeometry START" 
                 << " verbosity " << m_verbosity 
                 << " mm " << mm->getIndex()
                 ; 

    // when using --test eg PmtInBox or BoxInBox the mesh is fabricated in GGeoTest

    GParts* pts = mm->getParts(); assert(pts && "GMergedMesh with GEOCODE_ANALYTIC must have associated GParts, see GGeo::modifyGeometry "); 

    if(pts->getPrimBuffer() == NULL)
    {
        if(m_verbosity > 4) LOG(warning) << "OGeo::makeAnalyticGeometry GParts::close START " ; 
        pts->close();
        if(m_verbosity > 4) LOG(warning) << "OGeo::makeAnalyticGeometry GParts::close DONE " ; 
    }

    if(m_verbosity > 3) pts->fulldump("OGeo::makeAnalyticGeometry") ;

    //pts->setName("analytic");
    //pts->fulldump("OGeo::makeAnalyticGeometry") ;
    //pts->save("$TMP/OGeo_makeAnalyticGeometry");  // as here are now ~22 mm this just overwites..


    NPY<float>*     partBuf = pts->getPartBuffer(); assert(partBuf && partBuf->hasShape(-1,4,4));    // node buffer
    NPY<float>*     tranBuf = pts->getTranBuffer(); assert(tranBuf && tranBuf->hasShape(-1,3,4,4));  // transform triples (t,v,q) 
    NPY<float>*     planBuf = pts->getPlanBuffer(); assert(planBuf && planBuf->hasShape(-1,4));      // planes used for convex polyhedra such as trapezoid
    NPY<int>*       primBuf = pts->getPrimBuffer(); assert(primBuf && primBuf->hasShape(-1,4));      // prim
    NPY<unsigned>*  idBuf = mm->getAnalyticInstancedIdentityBuffer(); assert(idBuf && ( idBuf->hasShape(-1,4) || idBuf->hasShape(-1,1,4)));
     // PmtInBox yielding -1,1,4 ?

    unsigned numPrim = primBuf->getNumItems();
    unsigned numPart = partBuf->getNumItems();
    unsigned numTran = tranBuf->getNumItems();
    unsigned numPlan = planBuf->getNumItems();

    //assert( numPrim < 10 );  // expecting small number
    assert( numTran <= numPart ) ; 

    unsigned analytic_version = pts->getAnalyticVersion();

    OGeoStat stat(mm->getIndex(), numPrim, numPart, numTran, numPlan );
    m_stats.push_back(stat);

    if(m_verbosity > 2)
    LOG(info) 
                 << "OGeo::makeAnalyticGeometry " 
                 << stat.desc()
                 << " analytic_version " << analytic_version
                 ;

    optix::Geometry geometry = m_context->createGeometry();

    geometry->setPrimitiveCount( numPrim );
    geometry["primitive_count"]->setUint( numPrim );  // needed GPU side, for instanced offsets 
    geometry["analytic_version"]->setUint(analytic_version);

    optix::Program intersectProg = m_ocontext->createProgram("intersect_analytic.cu.ptx", "intersect") ;
    optix::Program boundsProg  =  m_ocontext->createProgram("intersect_analytic.cu.ptx", "bounds") ;

    geometry->setIntersectionProgram(intersectProg );
    geometry->setBoundingBoxProgram( boundsProg );

    assert(sizeof(int) == 4);
    optix::Buffer primBuffer = createInputUserBuffer<int>( primBuf,  4*4, "primBuffer"); 
    geometry["primBuffer"]->setBuffer(primBuffer);
    // hmm perhaps prim and id should be handled together ? 

    assert(sizeof(float) == 4);
    optix::Buffer partBuffer = createInputUserBuffer<float>( partBuf,  4*4*4, "partBuffer"); 
    geometry["partBuffer"]->setBuffer(partBuffer);

    assert(sizeof(optix::Matrix4x4) == 4*4*4);
    optix::Buffer tranBuffer = createInputUserBuffer<float>( tranBuf,  sizeof(optix::Matrix4x4), "tranBuffer"); 
    geometry["tranBuffer"]->setBuffer(tranBuffer);

    optix::Buffer identityBuffer = createInputBuffer<optix::uint4, unsigned int>( idBuf, RT_FORMAT_UNSIGNED_INT4, 1 , "identityBuffer"); 
    geometry["identityBuffer"]->setBuffer(identityBuffer);

    optix::Buffer planBuffer = createInputUserBuffer<float>( planBuf,  4*4, "planBuffer"); 
    geometry["planBuffer"]->setBuffer(planBuffer);

    // TODO: prismBuffer is misnamed it contains planes, TODO:migrate to use the planBuffer
    optix::Buffer prismBuffer = m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT);
    prismBuffer->setFormat(RT_FORMAT_FLOAT4);
    prismBuffer->setSize(5);
    geometry["prismBuffer"]->setBuffer(prismBuffer);

    if(m_verbosity > 2)
    LOG(warning) << "OGeo::makeAnalyticGeometry DONE" 
                 << " verbosity " << m_verbosity 
                 << " mm " << mm->getIndex()
                 ; 

    return geometry ; 
}


void OGeo::dumpStats(const char* msg)
{
    LOG(info) << msg << " num_stats " << m_stats.size() ; 
    for(unsigned i=0 ; i < m_stats.size() ; i++) std::cout << m_stats[i].desc() << std::endl ; 
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
    //
    // Hmm this is really treating each triangle as a primitive each with its own bounds...
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

    LOG(trace) << "OGeo::makeTriangulatedGeometry " 
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
        LOG(trace) << "OGeo::makeTriangulatedGeometry using FaceRepeatedInstancedIdentityBuffer"
                  << " friid items " << id->getNumItems() 
                  << " numITransforms*numFaces " << numITransforms*numFaces
                  ;

        assert( id->getNumItems() == numITransforms*numFaces );
   }
   else
   {
        id = mm->getFaceRepeatedIdentityBuffer();
        assert(id);
        LOG(trace) << "OGeo::makeTriangulatedGeometry using FaceRepeatedIdentityBuffer"
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


   if(m_verbosity > 2)
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
   unsigned int bit = buf->getNumItems(0,-1) ;  // (ifr,ito) -> size of all but the last dimension
   unsigned int nit = bit/fold ; 
   unsigned int mul = OConfig::getMultiplicity(format) ; // eg multiplicity of FLOAT4 is 4

   int buffer_id = buf->getBufferId() ;

   bool from_gl = buffer_id > -1 && reuse ;

   if(m_verbosity > 3 || strcmp(name,"tranBuffer") == 0)
   LOG(info)<<"OGeo::createInputBuffer [NPY<T>] "
            << " sh " << buf->getShapeString()
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

/*


2017-04-09 14:19:05.984 INFO  [4707966] [OGeo::createInputBuffer@687] OGeo::createInputBuffer [NPY<T>]  
     sh 1,2,4,4 
    fmt            FLOAT4 
    name       tranBuffer 
    bytes             128     1*2*4*4 = 32 floats, 32*4 = 128 bytes
      bit               8     1*2*4   
      nit               8     bit/fold(=1) <== CRUCIAL NUM-OF-float4 IN THE BUFFER 
      nel               4               <-- last dimension 
      mul               4 
     fold               1 
   sizeof(T)           16 
   sizeof(T)*nit      128 
         id            -1 
     from_gl            0

*/


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




template<typename T>
optix::Buffer OGeo::createInputUserBuffer(NPY<T>* src, unsigned elementSize, const char* name)
{
   return CreateInputUserBuffer(m_context, src, elementSize, name, m_verbosity);
}


template<typename T>
optix::Buffer OGeo::CreateInputUserBuffer(optix::Context& ctx, NPY<T>* src, unsigned elementSize, const char* name, unsigned verbosity)
{
    unsigned numBytes = src->getNumBytes() ;
    assert( numBytes % elementSize == 0 );
    unsigned size = numBytes/elementSize ; 

    if(verbosity > 2)
    LOG(info) << "OGeo::CreateInputUserBuffer"
              << " name " << name
              << " src shape " << src->getShapeString()
              << " numBytes " << numBytes
              << " elementSize " << elementSize
              << " size " << size 
              ;

    optix::Buffer buffer = ctx->createBuffer( RT_BUFFER_INPUT );

    buffer->setFormat( RT_FORMAT_USER );
    buffer->setElementSize(elementSize);
    buffer->setSize(size);

    memcpy( buffer->map(), src->getPointer(), numBytes );
    buffer->unmap();

    return buffer ; 
}


template
optix::Buffer OGeo::CreateInputUserBuffer<float>(optix::Context& ctx, NPY<float>* src, unsigned elementSize, const char* name, unsigned verbosity) ;





