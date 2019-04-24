
#include "OGeo.hh"
#include "OGeoStat.hh"
#include "OContext.hh"

#ifdef WITH_OPENGL
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#endif

#include <sstream>
#include <algorithm>
#include <iomanip>

#include <optix_world.h>

// opticks-
#include "Opticks.hh"
#include "OpticksConst.hh"

// optixrap-
#include "OContext.hh"
#include "OConfig.hh"
#include "OFormat.hh"

//#include "GGeo.hh"
//#include "GGeoBase.hh"
#include "GGeoLib.hh"
#include "GMergedMesh.hh"
#include "GParts.hh"

#include "PLOG.hh"
#include "BStr.hh"

// npy-
#include "NPY.hpp"
#include "NGPU.hpp"
#include "NSlice.hpp"
#include "GLMFormat.hpp"

#include "OConfig.hh"

/**

OGeo
=====

Prior to instancing 
---------------------

Simplest possible node tree::

   geometry_group 
       acceleration  
       geometry_instance 0
       geometry_instance 1
       ...            

1 to 1 mapping of GMergedMesh to "geometry_instance" 
each of which comprises one "geometry" with a single "material"
which refers to boundary lib properties lodged in the context by OBoundaryLib

Preparing for instancing
---------------------------

Transforms can only be contained in "group" so 
add top level group with another acceleration structure::

   group (top)
       acceleration
       geometry_group
             acceleration
             geometry_instance 0
             geometry_instance 1
                 

With instancing
------------------

::

     m_top (Group)
     acceleration

             m_global (GeometryGroup)
                 acceleration
                 geometry_instance 0
                 geometry_instance 1

             m_repeated (Group)
                 acceleration 

                 group 0
                     acceleration
                     xform_0 
                           repeated (GeometryGroup)
                     xform_1
                           repeated (GeometryGroup)
                     ...
 
                 group 1
                      acceleration
                      xform_0
                           repeated
                      ...


                  where repeated contains single gi (GeometryInstance) 


With instancing and ability to identify the intersected instance
---------------------------------------------------------------------

Need to be able to assign an index to each instance ...
means need a GeometryInstance beneath the xform ? 

The transform must be assigned exactly one child of type rtGroup, rtGeometryGroup, rtTransform, or rtSelector,


TODO:
Currently all the accelerations are using Sbvh/Bvh.
Investigate if container groups might be better as "pass through" 
NoAccel as the geometryGroup and groups they contain have all the 
geometry.

**/


const plog::Severity OGeo::LEVEL = debug ; 

const char* OGeo::BUILDER = "Sbvh" ; 
const char* OGeo::TRAVERSER = "Bvh" ; 


OGeo::OGeo(OContext* ocontext, Opticks* ok, GGeoLib* geolib, const char* builder, const char* traverser)
    : 
    m_ocontext(ocontext),
    m_ok(ok),
    m_gltf(ok->getGLTF()),
    m_geolib(geolib),
    m_builder(builder ? strdup(builder) : BUILDER),
    m_traverser(traverser ? strdup(traverser) : TRAVERSER),
    m_description(NULL),
    m_verbosity(m_ok->getVerbosity()),
    m_mmidx(0)
{
    init();
}

void OGeo::init()
{
    m_context = m_ocontext->getContext();
    m_global = m_context->createGeometryGroup();  
    m_repeated = m_context->createGroup();          // instanced geometry
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

    LOG(info) << "[ nmm " << nmm ;

    if(m_verbosity > 1) m_geolib->dump("OGeo::convert GGeoLib" ); 

    for(unsigned i=0 ; i < nmm ; i++) convertMergedMesh(i); 

    unsigned globalCount = m_global->getChildCount() ;
    unsigned repeatedCount = m_repeated->getChildCount() ;

    if(globalCount > 0)  // all Group and GeometryGroup need distinct acceleration structures
    {
         m_top->addChild(m_global);
         m_global->setAcceleration( makeAcceleration() );
    } 

    if(repeatedCount > 0)
    {
         m_top->addChild(m_repeated);
         m_repeated->setAcceleration( makeAcceleration() );
    } 

    m_top->setAcceleration( makeAcceleration() );

    if(m_verbosity > 0) dumpStats(); 
    LOG(info) << "] nmm " << nmm << " global " << globalCount << " repeated " << repeatedCount ; 
}

void OGeo::convertMergedMesh(unsigned i)
{
    m_mmidx = i ; 
    LOG(info) << "( " << i ; 

    GMergedMesh* mm = m_geolib->getMergedMesh(i); 

    bool raylod = m_ok->isRayLOD() ; 
    if(raylod) LOG(fatal) << " RayLOD enabled " ; 
    
    bool is_null = mm == NULL ; 
    bool is_skip = mm->isSkip() ;  
    bool is_empty = mm->isEmpty() ;  
    
    if( is_null || is_skip || is_empty )
    {
        LOG(error) << " not converting mesh " << i << " is_null " << is_null << " is_skip " << is_skip << " is_empty " << is_empty ; 
        return  ; 
    }

    if( i == 0 )         // global non-instanced geometry in slot 0
    {
        unsigned lod = 0u ;  
        optix::Material mat = makeMaterial();
        OGeometry* omm = makeOGeometry( mm, lod ); 

        unsigned instance_index = 0u ;  // so same code can run Instanced or not
        optix::GeometryInstance gi = makeGeometryInstance(omm, mat, instance_index );

        gi["primitive_count"]->setUint( 0u ); // not needed for non-instanced
        m_global->addChild(gi);
    }
    else        // repeated geometry
    {
        optix::Group group = makeRepeatedGroup(mm, raylod) ;
        group->setAcceleration( makeAcceleration() );
        m_repeated->addChild(group); 
    }
    LOG(info) << ") " << i ; 
}




/**
OGeo::makeRepeatedGroup
--------------------------

Geometry tree that allows instance identity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

          assembly        (Group)                1:1 with instanced merged mesh
             xform.0      (Transform)
               perxform   (GeometryGroup)
                 pergi    (GeometryInstance)       distinct pergi for every instance, with instance_index assigned  
                     omm  (Geometry)               the same omm and mat are child of all xform/perxform/pergi
                     mat  (Material) 
             xform.1       (Transform)
               perxform   (GeometryGroup)
                  pergi   (GeometryInstance)      
                     omm  (Geometry)
                     mat  (Material) 

             ... for all the many thousands of instances of repeated geometry ...


Why proliferate the *pergi* ? So can assign an instance index to it : ie know which PMT gets hit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* What other geo types can hold variables ? 
      
  * "Geometry" and "Material" can, but doesnt help for instance_index as only one of those
  
Could the perxform GeometryGroup be common to all ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NO, needs to be a separate GeometryGroup into which to place 
the distinct pergi GeometryInstance required for instanced identity   

Where to put the RayLOD Selector ? 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* NOTE THAT RAYLOD IS NOT CURRENTLY IN USE
* LOD : Level Of Detail 

  * level0 : most precise/expensive used for ray.origin inside instance sphere
  * level1 : cheaper alternative used for ray.origin outside instance sphere


The RayLOD idea is to switch geometry based on the distance from it, using 
radius of outermost solid origin centered bounding sphere with safety margin
see notes/issues/can-optix-selector-defer-expensive-csg.rst

Given that the same omm is used for all pergi... 
it would seem most appropriate to arrange the selector in common also, 
as all instances have the same simplified version of their geometry too..
BUT: selector needs to house 

*  Group contains : rtGroup, rtGeometryGroup, rtTransform, or rtSelector
*  Transform houses single child : rtGroup, rtGeometryGroup, rtTransform, or rtSelector   (NB not GeometryInstance)
*  GeometryGroup is a container for an arbitrary number of geometry instances, and must be assigned an Acceleration
*  Selector contains : rtGroup, rtGeometryGroup, rtTransform, and rtSelector

**/

optix::Group OGeo::makeRepeatedGroup(GMergedMesh* mm, bool raylod )
{
    const char geocode = m_ok->isXAnalytic() ? OpticksConst::GEOCODE_ANALYTIC : mm->getGeoCode();
    assert( geocode == OpticksConst::GEOCODE_TRIANGULATED || geocode == OpticksConst::GEOCODE_ANALYTIC || geocode == OpticksConst::GEOCODE_RTXTRIANGLES ) ;

    float instance_bounding_radius = mm->getBoundingRadiusCE(0) ; 

    NPY<float>* itransforms = mm->getITransformsBuffer();

    NSlice* islice = mm->getInstanceSlice(); 
    if(!islice) islice = new NSlice(0, itransforms->getNumItems()) ;

    unsigned int numTransforms = islice->count();
    assert(itransforms && numTransforms > 0);

    NPY<unsigned int>* ibuf = mm->getInstancedIdentityBuffer();
    unsigned int numIdentity = ibuf->getNumItems();

    assert(numIdentity % numTransforms == 0 && "expecting numIdentity to be integer multiple of numTransforms"); 
    unsigned int numSolids = numIdentity/numTransforms ;

    LOG(verbose) << "OGeo::makeRepeatedGroup"
              << " numTransforms " << numTransforms 
              << " numIdentity " << numIdentity  
              << " numSolids " << numSolids  
              << " islice " << islice->description() 
              << " instance_bounding_radius " << instance_bounding_radius
              ; 


    OGeometry* omm[2] ; 

    omm[0] = makeOGeometry( mm, 0u ); 
    omm[1] = raylod ? makeOGeometry( mm, 1u ) : NULL ; 

    optix::Material mat = makeMaterial();
    optix::Program visit ; 
    if(raylod)
    {
        visit = m_ocontext->createProgram("visit_instance.cu", "visit_instance");
        visit["instance_bounding_radius"]->setFloat( instance_bounding_radius*2.f );
    }


    optix::Group assembly = m_context->createGroup();
    assembly->setChildCount(islice->count());

    optix::Acceleration accel[2] ;
    accel[0] = makeAcceleration() ;  //  common accel for all instances as same geometry
    accel[1] = makeAcceleration() ;  //  NB accel is not created inside the loop 

    unsigned ichild = 0 ; 
    for(unsigned int i=islice->low ; i<islice->high ; i+=islice->step) //  CAUTION HEAVY LOOP eg 20k PMTs 
    {
        optix::Transform xform = m_context->createTransform();
        glm::mat4 m4 = itransforms->getMat4(i) ; 
        const float* tdata = glm::value_ptr(m4) ;  
        
        setTransformMatrix(xform, tdata); 
        assembly->setChild(ichild, xform);
        ichild++ ;
        unsigned instance_index = i ; 

        if(raylod == false)
        {
            optix::GeometryInstance pergi = makeGeometryInstance(omm[0], mat, instance_index); 
            optix::GeometryGroup perxform = makeGeometryGroup(pergi, accel[0] );    
            xform->setChild(perxform);  
        }
        else
        {
            optix::GeometryInstance gi[2] ; 
            gi[0] = makeGeometryInstance( omm[0] , mat, instance_index ); 
            gi[1] = makeGeometryInstance( omm[1] , mat, instance_index );  

            optix::GeometryGroup    gg[2] ; 
            gg[0] = makeGeometryGroup(gi[0], accel[0]);    
            gg[1] = makeGeometryGroup(gi[1], accel[1]);    
         
            optix::Selector selector = m_context->createSelector();
            selector->setChildCount(2) ; 
            selector->setChild(0, gg[0] );
            selector->setChild(1, gg[1] ); 
            selector->setVisitProgram( visit );           

            xform->setChild(selector);   
        }
    }
    return assembly ;
}



void OGeo::setTransformMatrix(optix::Transform& xform, const float* tdata ) 
{
    bool transpose = true ; 
    optix::Matrix4x4 m(tdata) ;
    xform->setMatrix(transpose, m.getData(), 0); 
    //dump("OGeo::setTransformMatrix", m.getData());
}


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
    LOG(verbose) << "OGeo::makeMaterial " 
               << " radiance_ray " << OContext::e_radiance_ray  
               << " propagate_ray " << OContext::e_propagate_ray  
               ; 

    optix::Material material = m_context->createMaterial();
    material->setClosestHitProgram(OContext::e_radiance_ray, m_ocontext->createProgram("material1_radiance.cu", "closest_hit_radiance"));
    material->setClosestHitProgram(OContext::e_propagate_ray, m_ocontext->createProgram("material1_propagate.cu", "closest_hit_propagate"));
    return material ; 
}


optix::GeometryGroup OGeo::makeGeometryGroup(optix::GeometryInstance gi, optix::Acceleration accel )
{
    optix::GeometryGroup gg = m_context->createGeometryGroup();
    gg->addChild(gi);
    gg->setAcceleration( accel );
    return gg ;
}



optix::GeometryInstance OGeo::makeGeometryInstance(OGeometry* ogeom, optix::Material mat, unsigned instance_index)
{
    optix::GeometryInstance gi = m_context->createGeometryInstance() ;
    gi->setMaterialCount(1); 
    gi->setMaterial(0, mat ); 

    if( ogeom->g.get() != NULL )
    {
        gi->setGeometry( ogeom->g ); 
    }
#if OPTIX_VERSION >= 60000 
    else if ( ogeom->gt.get() != NULL )
    {
        gi->setGeometryTriangles( ogeom->gt ); 
    }
#endif  
    else
    {
        LOG(fatal) << " given OGeometry instance holding no geometry " ; 
        assert(0);  
    }
    gi["instance_index"]->setUint(instance_index);  
    return gi ;
}


/**
OGeo::makeOGeometry : creating the OptiX GPU geometry
-------------------------------------------------------

* NB --xanalytic option switches to analytic geometry, ignoring the 'T' or 'A' geocode of the mergedmesh 

The OGeo::OGeometry internal struct is returned to enable optix::Geometry and optix::GeometryTriangles
to be handled uniformly.  Initially tried returning optix::GeometryInstance to effect such a
uniform jandling but that was too slow as it forced recreation of optix geometry 
for every instance.

**/


OGeo::OGeometry* OGeo::makeOGeometry(GMergedMesh* mergedmesh, unsigned lod)
{
    OGeometry* ogeom = new OGeometry ; 
    
    const char geocode = m_ok->isXAnalytic() ? OpticksConst::GEOCODE_ANALYTIC : mergedmesh->getGeoCode() ;

    LOG(LEVEL) << "geocode " << geocode ; 

    if(geocode == OpticksConst::GEOCODE_TRIANGULATED)
    {
        ogeom->g = makeTriangulatedGeometry(mergedmesh, lod);
    }
    else if(geocode == OpticksConst::GEOCODE_ANALYTIC)
    {
        ogeom->g = makeAnalyticGeometry(mergedmesh, lod);
    }
    else if(geocode == OpticksConst::GEOCODE_RTXTRIANGLES)
    {
        ogeom->gt = makeRTXTrianglesGeometry(mergedmesh, lod);
    }
    else
    {
        LOG(fatal) << "geocode must be triangulated or analytic, not [" << (char)geocode  << "]" ;
        assert(0);
    }
    return ogeom ; 
}


optix::Geometry OGeo::makeAnalyticGeometry(GMergedMesh* mm, unsigned lod)
{
    m_lodidx = lod ; 

    if(m_verbosity > 2)
    LOG(warning) << "OGeo::makeAnalyticGeometry START" 
                 << " verbosity " << m_verbosity 
                 << " lod " << lod
                 << " mm " << mm->getIndex()
                 ; 

    // when using --test eg PmtInBox or BoxInBox the mesh is fabricated in GGeoTest

    GParts* pts = mm->getParts(); assert(pts && "GMergedMesh with GEOCODE_ANALYTIC must have associated GParts, see GGeo::modifyGeometry "); 

    if(pts->getPrimBuffer() == NULL)
    {
        LOG(debug) << "( GParts::close " ; 
        pts->close();
        LOG(debug) << ") GParts::close " ; 
    }
    else
    {
        LOG(debug) << " skip GParts::close " ; 
    }
    
    LOG(info) << "pts: " << pts->desc() ; 

    if(m_verbosity > 3 || m_ok->hasOpt("dbganalytic")) pts->fulldump("--dbganalytic", 10) ;

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

    unsigned int numVolumes = mm->getNumVolumes();
    unsigned int numVolumesSelected = mm->getNumVolumesSelected();

    if( pts->isNodeTree() )
    {
        bool match = numPrim == numVolumes ;
        if(!match)
        {
            LOG(fatal) << " NodeTree : MISMATCH (numPrim != numVolumes) "
                       << " numVolumes " << numVolumes 
                       << " numVolumesSelected " << numVolumesSelected 
                       << " numPrim " << numPrim 
                       << " numPart " << numPart 
                       << " numTran " << numTran 
                       << " numPlan " << numPlan 
                       ; 
        }
        //assert( match && "NodeTree Sanity check failed " );
        // hmm tgltf-;tgltf-- violates this ?
    }


    //assert( numPrim < 10 );  // expecting small number
    assert( numTran <= numPart ) ; 

    unsigned analytic_version = pts->getAnalyticVersion();

    OGeoStat stat(mm->getIndex(), numPrim, numPart, numTran, numPlan );
    m_stats.push_back(stat);

    if(m_verbosity > 2)
    LOG(info) 
                 << stat.desc()
                 << " analytic_version " << analytic_version
                 ;

    optix::Geometry geometry = m_context->createGeometry();

    assert( numPrim >= 1 );
    geometry->setPrimitiveCount( lod > 0 ? 1 : numPrim );  // lazy lod, dont change buffers, just ignore all but the 1st prim for lod > 0

    geometry["primitive_count"]->setUint( numPrim );       // needed GPU side, for instanced offset into buffers 
    geometry["analytic_version"]->setUint(analytic_version);

    optix::Program intersectProg = m_ocontext->createProgram("intersect_analytic.cu", "intersect") ;
    optix::Program boundsProg  =  m_ocontext->createProgram("intersect_analytic.cu", "bounds") ;

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


#if OPTIX_VERSION >= 60000
optix::GeometryTriangles OGeo::makeRTXTrianglesGeometry(GMergedMesh* mm, unsigned lod)
{
    assert(0);
    optix::GeometryTriangles geotri = m_context->createGeometryTriangles();
    return geotri  ; 
}
#endif


/**
OGeo::makeTriangulatedGeometry
---------------------------------

Index buffer items are the indices of every triangle vertex, so divide by 3 to get faces 
and use folding by 3 in createInputBuffer.
    
For instanced geometry this just sees (for DYB near) the 5 solids of the repeated instance 
and numFaces is the sum of the face counts of those, and numITransforms is 672
    
In order to provide identity to the instances need to repeat the iidentity to the triangles
    
* Hmm this is really treating each triangle as a primitive each with its own bounds...

**/

optix::Geometry OGeo::makeTriangulatedGeometry(GMergedMesh* mm, unsigned lod)
{
    m_lodidx = lod ; 

    optix::Geometry geometry = m_context->createGeometry();
    geometry->setIntersectionProgram(m_ocontext->createProgram("TriangleMesh.cu", "mesh_intersect"));
    geometry->setBoundingBoxProgram(m_ocontext->createProgram("TriangleMesh.cu", "mesh_bounds"));

    unsigned numVolumes = mm->getNumVolumes();
    unsigned numFaces = mm->getNumFaces();
    unsigned numITransforms = mm->getNumITransforms();
    unsigned numFaces0 = mm->getNodeInfo(0).x ; 

    LOG(debug) 
              << " lod " << lod
              << " mmIndex " << mm->getIndex() 
              << " numFaces (PrimitiveCount) " << numFaces
              << " numFaces0 (Outermost) " << numFaces0
              << " numVolumes " << numVolumes
              << " numITransforms " << numITransforms 
              ;
             
    geometry->setPrimitiveCount(lod > 0 ? numFaces0 : numFaces ); // lazy LOD, ie dont change buffer, just ignore most of it for lod > 0 

    geometry["primitive_count"]->setUint( numFaces );  // needed for instanced offsets into buffers, so must describe the buffer, NOT the intent 

    GBuffer* id = NULL ; 
    if(numITransforms > 0)  //  formerly 0   : HUH: perhaps should be 1,  always using friid even for globals ?
    {
        id = mm->getFaceRepeatedInstancedIdentityBuffer();
        assert(id);
        LOG(verbose) << "using FaceRepeatedInstancedIdentityBuffer" << " friid items " << id->getNumItems() << " numITransforms*numFaces " << numITransforms*numFaces ; 
        assert( id->getNumItems() == numITransforms*numFaces );
   }
   else
   {
        id = mm->getFaceRepeatedIdentityBuffer();
        assert(id);
        LOG(verbose) << "using FaceRepeatedIdentityBuffer" << " frid items " << id->getNumItems() << " numFaces " << numFaces ; 
        assert( id->getNumItems() == numFaces );
   }  

   optix::Buffer identityBuffer = createInputBuffer<optix::uint4>( id, RT_FORMAT_UNSIGNED_INT4, 1 , "identityBuffer"); 
   geometry["identityBuffer"]->setBuffer(identityBuffer);

   /**
    *reuse* was an unsuccessful former attempt to "purloin the OpenGL buffers" avoid duplicating geometry info between OpenGL and OptiX
    setting reuse to true causes OptiX/OpenGL launch failure : bad enum 
    **/

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

const char* OGeo::getContextName() const 
{
    std::stringstream ss ; 
    ss << "OGeo"
       << m_mmidx 
       << "-"
       << m_lodidx 
        ; 
      
    std::string name = ss.str();  
    return strdup(name.c_str()); 
}


template <typename T>
optix::Buffer OGeo::createInputBuffer(GBuffer* buf, RTformat format, unsigned int fold, const char* name, bool reuse)
{
   //TODO: eliminate use of this, moving to NPY buffers instead

   unsigned int bytes = buf->getNumBytes() ;

   unsigned int bit = buf->getNumItems() ; 
   unsigned int nit = bit/fold ; 
   unsigned int nel = buf->getNumElements();
   unsigned int mul = OFormat::Multiplicity(format) ;

   int buffer_id = buf->getBufferId() ;


   if(m_verbosity > 2)
   LOG(info)<<"OGeo::createInputBuffer [GBuffer]"
            << " fmt " << std::setw(20) << OFormat::FormatName(format)
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
        unsigned num_bytes = buf->getNumBytes() ;
        memcpy( buffer->map(), buf->getPointer(), num_bytes );
        buffer->unmap();

        NGPU::GetInstance()->add(num_bytes, name, getContextName() , "cibGBuf" ); 
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
   unsigned int mul = OFormat::Multiplicity(format) ; // eg multiplicity of FLOAT4 is 4

   int buffer_id = buf->getBufferId() ;

   bool from_gl = buffer_id > -1 && reuse ;

   if(m_verbosity > 3 || strcmp(name,"tranBuffer") == 0)
   LOG(info)<<"OGeo::createInputBuffer [NPY<T>] "
            << " sh " << buf->getShapeString()
            << " fmt " << std::setw(20) << OFormat::FormatName(format)
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
        unsigned num_bytes =  buf->getNumBytes() ;
        memcpy( buffer->map(), buf->getPointer(), num_bytes );
        buffer->unmap();
        NGPU::GetInstance()->add(num_bytes, name, getContextName(), "cibNPY" ); 
   } 

   return buffer ; 
}


template<typename T>
optix::Buffer OGeo::createInputUserBuffer(NPY<T>* src, unsigned elementSize, const char* name)
{
   const char* ctxname = getContextName();
   return CreateInputUserBuffer(m_context, src, elementSize, name, ctxname, m_verbosity);
}


template<typename T>
optix::Buffer OGeo::CreateInputUserBuffer(optix::Context& ctx, NPY<T>* src, unsigned elementSize, const char* name, const char* ctxname_informational, unsigned verbosity)
{
    unsigned numBytes = src->getNumBytes() ;
    assert( numBytes % elementSize == 0 );
    unsigned size = numBytes/elementSize ; 

    if(verbosity > 2)
    LOG(info) << "OGeo::CreateInputUserBuffer"
              << " name " << name
              << " ctxname " << ctxname_informational
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

    NGPU::GetInstance()->add(numBytes, name, ctxname_informational, "ciubNPY" ); 

    return buffer ; 
}


template
optix::Buffer OGeo::CreateInputUserBuffer<float>(optix::Context& ctx, NPY<float>* src, unsigned elementSize, const char* name, const char* ctxname, unsigned verbosity) ;




