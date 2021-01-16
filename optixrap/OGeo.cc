/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

 
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
#include "OGeometry.hh"



#include "GGeo.hh"
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
OGeo Details
-----------------

Table 2, OptiX Manual
~~~~~~~~~~~~~~~~~~~~~~~~~

=================   ================================   =========================
Parent Node Type     Child Node Types                   Associated Node Types
=================   ================================   =========================
Geometry               None                               Material
Acceleration           None                                     
GeometryInstance       Geometry                           Material
GeometryGroup          GeometryInstance                   Acceleration
Transform              GeometryGroup         
Selector               Transform
Group                  GeometryGroup                      Acceleration    
=================   ================================   =========================

*  Group contains : rtGroup, rtGeometryGroup, rtTransform, or rtSelector
*  Transform houses single child : rtGroup, rtGeometryGroup, rtTransform, or rtSelector   (NB not GeometryInstance)
*  GeometryGroup is a container for an arbitrary number of geometry instances, and must be assigned an Acceleration
*  Selector contains : rtGroup, rtGeometryGroup, rtTransform, and rtSelector



Geometry tree that allows instance identity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

JUNO has 6 repeated pieces of geometry.  
The two different types of photomultiplier tubes (PMTs) are 
by far the most prolific with 20k of one type (20inch) 
and 36k of another (3inch)

The geometry tree follows that show in OptiX 6.0.0 manual Fig 3.4 x6 

::

    m_top                  (Group)             m_top_accel
       ggg                 (GeometryGroup)        m_ggg_accel           global non-instanced geometry from merged mesh 0  
          ggi              (GeometryInstance)        

       assembly.0          (Group)                m_assembly_accel      1:1 with instanced merged mesh (~6 of these for JUNO)

             xform.0       (Transform)                                  (at most 20k/36k different transforms)
               perxform    (GeometryGroup)
                  accel[0]                            m_instance_accel  common accel within each assembly 
                  pergi    (GeometryInstance)                           distinct pergi for every instance, with instance_index assigned  
                     omm   (Geometry)                                   the same omm and mat are child of all xform/perxform/pergi
                     mat   (Material) 

             xform.1       (Transform)
               perxform    (GeometryGroup)
                  pergi    (GeometryInstance)      
                  accel[0]
                     omm   (Geometry)
                     mat   (Material) 

             ... for all the many thousands of instances of repeated geometry ...


       assembly.1          (Group)                  (order ~6 repeated assemblies for JUNO)
            xform.0  
            ... just like above ...



* transforms can only be contained in "group" or another transform so add top level group with 
  another acceleration structure

* transforms must be assigned exactly one child of type rtGroup, rtGeometryGroup, rtTransform, or rtSelector,


Alternate Tree Layout
~~~~~~~~~~~~~~~~~~~~~~~~
   
     (Group)
       (Transform)
          (GeometryGroup)
              (GeometryInstance)
                  (Geometry)
                   (Material)




OptiX 7 terminology change
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OptiX 7 changes terminology in a way which may inform 
concerning which trees can be handled in RT cores

* Geometry Group -> Geometry AS (only primitives)
* Group -> Instance AS
* Transform -> just input to Instance AS at build


Why proliferate the *pergi* ? So can assign an instance index to it : ie know which PMT gets hit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Need to assign an index to each instance means need a GeometryInstance beneath the xform ? 

* "Geometry" and "Material" can also hold variables, but that doesnt help for instance_index 
   as there is only one geometry and material instance for each assembly

  
Could the perxform GeometryGroup be common to all ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NO, needs to be a separate GeometryGroup into which to place 
the distinct pergi GeometryInstance required for instanced identity   

Where to put the RayLOD Selector ? RAYLOD IS NOT CURRENTLY IN USE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* LOD : Level Of Detail 

  * level0 : most precise/expensive used for ray.origin inside instance sphere
  * level1 : cheaper alternative used for ray.origin outside instance sphere

The RayLOD idea is to switch geometry based on the distance from it, using 
radius of outermost solid origin centered bounding sphere with safety margin
see notes/issues/can-optix-selector-defer-expensive-csg.rst

Given that the same omm is used for all pergi... 
it would seem most appropriate to arrange the selector in common also, 
as all instances have the same simplified version of their geometry too..

TODO:
Currently all the accelerations are using Sbvh/Bvh.
Investigate if container groups might be better as "pass through" 
NoAccel as the geometryGroup and groups they contain have all the 
geometry.

**/


const plog::Severity OGeo::LEVEL = PLOG::EnvLevel("OGeo", "DEBUG") ; 


const char* OGeo::ACCEL = "Sbvh" ; 

OGeo::OGeo(OContext* ocontext, Opticks* ok, GGeo* ggeo )
    : 
    m_ocontext(ocontext),
    m_context(m_ocontext->getContext()),
    m_ok(ok),
    m_ggeo(ggeo),
    m_geolib(ggeo->getGeoLib()),
    m_verbosity(m_ok->getVerbosity()),
    m_mmidx(0),
    m_top_accel(ACCEL),
    m_ggg_accel(ACCEL),
    m_assembly_accel(ACCEL),
    m_instance_accel(ACCEL)
{
    init();
}

void OGeo::init()
{
    const char* accel = m_ok->getAccel(); 

    std::vector<std::string> elem ; 
    BStr::split(elem, accel, ','); 
    unsigned nelem = elem.size(); 

    if(nelem > 0) m_top_accel = strdup(elem[0].c_str()) ; 
    if(nelem > 1) m_ggg_accel = strdup(elem[1].c_str()) ; 
    if(nelem > 2) m_assembly_accel = strdup(elem[2].c_str()) ; 
    if(nelem > 3) m_instance_accel = strdup(elem[3].c_str()) ; 

    setTopGroup(m_ocontext->getTopGroup());

    initWayControl(); 

    LOG(info) << description() ; 
}


void OGeo::initWayControl()
{
    int node     = m_ggeo->getFirstNodeIndexForPVName();   // --pvname pInnerWater 
    int boundary = m_ggeo->getSignedBoundary() ;           // --boundary Water///Acrylic  

    optix::int4 way_control = optix::make_int4(node,boundary,0,0); // HMM: need GGeo to set these
    LOG(LEVEL) 
        << " way_control.x (node) " << way_control.x 
        << " way_control.y (boundary) " << way_control.y
        << " way_control.z " << way_control.z 
        << " way_control.w " << way_control.w 
        ;
    m_context["way_control"]->setInt(way_control); 
} 



void OGeo::setTopGroup(optix::Group top)
{
    m_top = top ; 
}

std::string OGeo::description() const 
{
    std::stringstream ss ; 
    ss << "OGeo "
       << " top " << m_top_accel 
       << " ggg " << m_ggg_accel
       << " assembly " << m_assembly_accel
       << " instance " << m_instance_accel
       ;
    return ss.str(); 
}

void OGeo::convert()
{
    m_geolib->dump("OGeo::convert"); 

    unsigned int nmm = m_geolib->getNumMergedMesh();

    LOG(info) << "[ nmm " << nmm ;

    if(m_verbosity > 1) m_geolib->dump("OGeo::convert GGeoLib" ); 

    for(unsigned i=0 ; i < nmm ; i++) 
    {
        convertMergedMesh(i); 
    }

    m_top->setAcceleration( makeAcceleration(m_top_accel, false) );

    if(m_verbosity > 0) dumpStats(); 

    LOG(info) << "] nmm " << nmm  ; 
}

void OGeo::convertMergedMesh(unsigned i)
{
    LOG(LEVEL) << "( " << i  ; 
    m_mmidx = i ; 

    GMergedMesh* mm = m_geolib->getMergedMesh(i); 

    bool raylod = m_ok->isRayLOD() ; 
    if(raylod) LOG(fatal) << " RayLOD enabled " ; 
    assert( raylod == false ); 
    
    bool is_null = mm == NULL ; 
    bool is_skip = mm->isSkip() ;  
    bool is_empty = mm->isEmpty() ;  
    
    if( is_null || is_skip || is_empty )
    {
        LOG(error) << " not converting mesh " << i << " is_null " << is_null << " is_skip " << is_skip << " is_empty " << is_empty ; 
        return  ; 
    }

    unsigned numInstances = 0 ; 
    if( i == 0 )   // global non-instanced geometry in slot 0
    {
        optix::GeometryGroup ggg = makeGlobalGeometryGroup(mm);
        m_top->addChild(ggg); 
        numInstances = 1 ; 
    }
    else           // repeated geometry
    {
        optix::Group assembly = makeRepeatedAssembly(mm) ;
        assembly->setAcceleration( makeAcceleration(m_assembly_accel, false) );
        numInstances = assembly->getChildCount() ; 
        m_top->addChild(assembly); 
    }
    LOG(LEVEL) << ") " << i << " numInstances " << numInstances ; 
}

optix::GeometryGroup OGeo::makeGlobalGeometryGroup(GMergedMesh* mm)
{
    int dbgmm =  m_ok->getDbgMM() ; 
    if(dbgmm == 0) mm->dumpVolumesSelected("OGeo::makeGlobalGeometryGroup [--dbgmm 0] "); 

    optix::Material mat = makeMaterial();
    OGeometry* omm = makeOGeometry( mm ); 

    unsigned instance_index = 0u ;  // so same code can run Instanced or not
    optix::GeometryInstance ggi = makeGeometryInstance(omm, mat, instance_index );
    ggi["primitive_count"]->setUint( 0u );  // non-instanced
    ggi["repeat_index"]->setUint( mm->getIndex() );  // non-instanced

    optix::Acceleration accel = makeAcceleration(m_ggg_accel, false) ;
    optix::GeometryGroup ggg = makeGeometryGroup(ggi, accel );    

#if OPTIX_VERSION_MAJOR >= 6
    RTinstanceflags instflags = RT_INSTANCE_FLAG_DISABLE_ANYHIT ;  
    ggg->setFlags(instflags);
#endif

    return ggg ; 
}


/**
OGeo::makeRepeatedAssembly
---------------------------

Invoked only from OGeo::convertMergedMesh.


**/

optix::Group OGeo::makeRepeatedAssembly(GMergedMesh* mm)
{
    bool raylod = false ; 
    unsigned mmidx = mm->getIndex(); 
    unsigned imodulo = m_ok->getInstanceModulo( mmidx ); 

    LOG(LEVEL) 
         << " mmidx " << mmidx 
         << " imodulo " << imodulo
         ;


    NPY<float>* itransforms = mm->getITransformsBuffer();

    NSlice* islice = mm->getInstanceSlice(); 
    if(!islice) islice = new NSlice(0, itransforms->getNumItems()) ;

    unsigned int numTransforms = islice->count();
    assert(itransforms && numTransforms > 0);

    NPY<unsigned int>* ibuf = mm->getInstancedIdentityBuffer();
    unsigned int numIdentity = ibuf->getNumItems();

    assert(numIdentity % numTransforms == 0 && "expecting numIdentity to be integer multiple of numTransforms"); 
    unsigned int numSolids = numIdentity/numTransforms ;

    LOG(LEVEL)
        << " numTransforms " << numTransforms 
        << " numIdentity " << numIdentity  
        << " numSolids " << numSolids  
        << " islice " << islice->description() 
        ; 


    OGeometry* omm[2] ; 

    omm[0] = makeOGeometry( mm ); 
    //omm[1] = raylod ? makeOGeometry( mm, 1u ) : NULL ; 
    omm[1] = NULL ; 

    optix::Material mat = makeMaterial();

#ifdef OLD_LOD
    optix::Program visit ; 
    if(raylod)
    {
        visit = m_ocontext->createProgram("visit_instance.cu", "visit_instance");
        float instance_bounding_radius = mm->getBoundingRadiusCE(0) ; 
        visit["instance_bounding_radius"]->setFloat( instance_bounding_radius*2.f );
    }
#endif

   
    unsigned count(0); 
    if(imodulo == 0u)
    {
        count = islice->count() ; 
    }
    else
    {
        for(unsigned int i=islice->low ; i<islice->high ; i+=islice->step) //  CAUTION HEAVY LOOP eg 20k PMTs 
        {
            if( i % imodulo != 0u ) continue ;   
            count++ ; 
        }
    }


    optix::Group assembly = m_context->createGroup();
    assembly->setChildCount( count );

    optix::Acceleration accel[2] ;
    accel[0] = makeAcceleration(m_instance_accel, false) ;  //  common accel for all instances as same geometry
    accel[1] = makeAcceleration(m_instance_accel, false) ;  //  NB accel is not created inside the below instance loop 

    unsigned ichild = 0 ; 
    for(unsigned int i=islice->low ; i<islice->high ; i+=islice->step) //  CAUTION HEAVY LOOP eg 20k PMTs 
    {
        if( imodulo > 0u && i % imodulo != 0u ) continue ;     // modulo scaledown for debugging

        optix::Transform xform = m_context->createTransform();
        glm::mat4 m4 = itransforms->getMat4(i) ; 
        const float* tdata = glm::value_ptr(m4) ;  
        
        setTransformMatrix(xform, tdata); 
        assembly->setChild(ichild, xform);
        ichild++ ;
        unsigned instance_index = i ; 

        if(raylod == false)
        {
            /*    
             assembly             (Group) 
                xform             (Transform)
                   perxform       (GeometryGroup)
                       pergi      (GeometryInstance)  
                       accel[0]   (Acceleration)   
            */  
            optix::GeometryInstance pergi = makeGeometryInstance(omm[0], mat, instance_index); 
            optix::GeometryGroup perxform = makeGeometryGroup(pergi, accel[0] );    
            xform->setChild(perxform);  

#if OPTIX_VERSION_MAJOR >= 6
            RTinstanceflags instflags = RT_INSTANCE_FLAG_DISABLE_ANYHIT ;  
            perxform->setFlags(instflags);
#endif
        }
#ifdef OLD_LOD
        else
        {
            assert(0);  
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
#endif
    }

    assert( ichild == count );

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


optix::Acceleration OGeo::makeAcceleration(const char* accel, bool accel_props)
{

    LOG(debug)
        << " accel " << accel 
        << " accel_props " << accel_props
        ; 
 
    optix::Acceleration acceleration = m_context->createAcceleration(accel);
    if(accel_props == true)
    {
        acceleration->setProperty( "vertex_buffer_name", "vertexBuffer" );
        acceleration->setProperty( "index_buffer_name", "indexBuffer" );
    }
    return acceleration ; 
}

optix::Material OGeo::makeMaterial()
{
    LOG(verbose) 
        << " radiance_ray " << OContext::e_radiance_ray  
        << " propagate_ray " << OContext::e_propagate_ray  
        ; 

    LOG(LEVEL) << "[" ; 
    optix::Material material = m_context->createMaterial();
    material->setClosestHitProgram(OContext::e_radiance_ray, m_ocontext->createProgram("material1_radiance.cu", "closest_hit_radiance"));
    material->setClosestHitProgram(OContext::e_propagate_ray, m_ocontext->createProgram("material1_propagate.cu", "closest_hit_propagate"));
    LOG(LEVEL) << "]" ; 
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
uniform handling but that was too slow as it forced recreation of optix geometry 
for every instance.

**/


OGeometry* OGeo::makeOGeometry(GMergedMesh* mergedmesh)
{

    OGeometry* ogeom = new OGeometry ; 

/*
    int rtxmode = m_ok->getRTX(); 

    char ugeocode ; 

    if( m_ok->isXAnalytic() )
    {
         ugeocode = OpticksConst::GEOCODE_ANALYTIC ;  
    } 
    else if(  m_ok->isXGeometryTriangles() || rtxmode == 2 )
    {
         ugeocode = OpticksConst::GEOCODE_GEOMETRYTRIANGLES ;  
    }
    else
    {
         ugeocode = mergedmesh->getGeoCode() ;  
    }
*/
    char ugeocode  = mergedmesh->getCurrentGeoCode(); 

    LOG(LEVEL) << "ugeocode [" << (char)ugeocode << "]" ; 

    if(ugeocode == OpticksConst::GEOCODE_TRIANGULATED )
    {
        ogeom->g = makeTriangulatedGeometry(mergedmesh);
    }
    else if(ugeocode == OpticksConst::GEOCODE_ANALYTIC)
    {
        ogeom->g = makeAnalyticGeometry(mergedmesh);
    }
    else if(ugeocode == OpticksConst::GEOCODE_GEOMETRYTRIANGLES)
    {
#if OPTIX_VERSION_MAJOR >= 6
        ogeom->gt = makeGeometryTriangles(mergedmesh);
#else
        assert(0 && "Require at least OptiX 6.0.0 to use GeometryTriangles "); 
#endif
    }
    else
    {
        LOG(fatal) << "geocode must be triangulated or analytic, not [" << (char)ugeocode  << "]" ;
        assert(0);
    }


#if OPTIX_VERSION_MAJOR >= 6 
    LOG(verbose) << " DISABLE_ANYHIT " ; 

    RTgeometryflags flags = RT_GEOMETRY_FLAG_DISABLE_ANYHIT ;  
    if(ogeom->isGeometry())
    {
        ogeom->g->setFlags( flags ); 
    }
    else if( ogeom->isGeometryTriangles())
    {
        unsigned int material_index = 0u ;   
        ogeom->gt->setFlagsPerMaterial( material_index, flags ); 
    }
#endif

    return ogeom ; 
}


/**
OGeo::makeAnalyticGeometry
----------------------------

The GParts instance that this operates from will usually 
have been concatenated from multiple other GParts instances, 
one for each NCSG solid.  GParts concatenation happens during 
GMergedMesh formation in GMergedMesh::mergeVolumeAnalytic.

For repeated geometry note how all bar one of the geometry buffers 
are small. Only the idBuf is large and usage GPU side requires 
use of the instance_index. 

**/

optix::Geometry OGeo::makeAnalyticGeometry(GMergedMesh* mm)
{
    bool dbgmm = m_ok->getDbgMM() == int(mm->getIndex()) ;  
    bool dbganalytic = m_ok->hasOpt("dbganalytic") ; 


    if(m_verbosity > 2 || dbgmm)
    LOG(info) 
         << "["
         << " verbosity " << m_verbosity 
         << " mm " << mm->getIndex()
         ; 

    // when using --test eg PmtInBox or BoxInBox the mesh is fabricated in GGeoTest

    GParts* pts = mm->getParts(); assert(pts && "GMergedMesh with GEOCODE_ANALYTIC must have associated GParts, see GGeo::modifyGeometry "); 

    if(pts->getPrimBuffer() == NULL)
    {
        LOG(LEVEL) << "( GParts::close " ; 
        pts->close();
        LOG(LEVEL) << ") GParts::close " ; 
    }
    else
    {
        LOG(LEVEL) << " skip GParts::close " ; 
    }
    
    LOG(LEVEL) << "mm " << mm->getIndex() 
              << " verbosity: " << m_verbosity   
              << ( dbgmm ? " --dbgmm " : " " )
              << ( dbganalytic ? " --dbganalytic " : " " )
              << " pts: " << pts->desc() 
              ;  
 
    if(dbgmm)
    {
        LOG(fatal) << "dumping as instructed by : --dbgmm " << m_ok->getDbgMM() ;   
        mm->dumpVolumesSelected("OGeo::makeAnalyticGeometry"); 
    }


    if(m_verbosity > 3 || dbganalytic || dbgmm ) pts->fulldump("--dbganalytic/--dbgmm", 10) ;

    NPY<float>*     partBuf = pts->getPartBuffer(); assert(partBuf && partBuf->hasShape(-1,4,4));    // node buffer
    NPY<float>*     tranBuf = pts->getTranBuffer(); assert(tranBuf && tranBuf->hasShape(-1,3,4,4));  // transform triples (t,v,q) 
    NPY<float>*     planBuf = pts->getPlanBuffer(); assert(planBuf && planBuf->hasShape(-1,4));      // planes used for convex polyhedra such as trapezoid
    NPY<int>*       primBuf = pts->getPrimBuffer(); assert(primBuf && primBuf->hasShape(-1,4));      // prim

    unsigned numPrim = primBuf->getNumItems();

    NPY<float>* itransforms = mm->getITransformsBuffer(); assert(itransforms && itransforms->hasShape(-1,4,4) ) ;
    unsigned numInstances = itransforms->getNumItems(); 
    NPY<unsigned>*  idBuf = mm->getInstancedIdentityBuffer();   assert(idBuf); 
    LOG(LEVEL) 
        << " mmidx " << mm->getIndex()
        << " numInstances " << numInstances 
        << " numPrim " << numPrim
        << " idBuf " << idBuf->getShapeString()
        ;
           
    if( mm->getIndex() > 0 )  // volume level buffers do not honour selection unless using globalinstance
    {
        assert(idBuf->hasShape(numInstances,numPrim,4)); 
    }



    unsigned numPart = partBuf->getNumItems();
    unsigned numTran = tranBuf->getNumItems();
    unsigned numPlan = planBuf->getNumItems();

    unsigned numVolumes = mm->getNumVolumes();
    unsigned numVolumesSelected = mm->getNumVolumesSelected();

    if( pts->isNodeTree() )
    {
        bool match = numPrim == numVolumes ;
        if(!match)
        {
            LOG(fatal) 
                << " NodeTree : MISMATCH (numPrim != numVolumes) "
                << " (this happens when using --csgskiplv) " 
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

    bool someprim = numPrim >= 1 ; 
    if(!someprim)
        LOG(fatal) 
            << " someprim fails " 
            << " mm.index " << mm->getIndex()
            << " numPrim " << numPrim 
            << " numPart " << numPart 
            << " numTran " << numTran 
            << " numPlan " << numPlan 
            ;

    assert( someprim );
#ifdef OLD_LOD
    geometry->setPrimitiveCount( lod > 0 ? 1 : numPrim );  // lazy lod, dont change buffers, just ignore all but the 1st prim for lod > 0
#else
    geometry->setPrimitiveCount( numPrim ); 
#endif


    geometry["primitive_count"]->setUint( numPrim );       // needed GPU side, for instanced offset into buffers 
    geometry["repeat_index"]->setUint( mm->getIndex() );  // ridx
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

    if(m_verbosity > 2 || dbgmm)
    LOG(info)
        << "]" 
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

/**
OGeo::makeGeometryTriangles
-----------------------------

Cannot use int3 index format::

    what():  Invalid context (Details: Function "RTresult _rtContextValidate(RTcontext)" caught exception: 
       Validation error: GeometryTriangles has invalid index format, must be RT_FORMAT_UNSIGNED_INT3, or RT_FORMAT_UNSIGNED_SHORT3)


**/

#if OPTIX_VERSION >= 60000
optix::GeometryTriangles OGeo::makeGeometryTriangles(GMergedMesh* mm)
{
    unsigned numFaces = mm->getNumFaces();
    unsigned numVertices = mm->getNumVertices(); 

    GBuffer* vtx = mm->getVerticesBuffer() ;
    GBuffer* rib = mm->getAppropriateRepeatedIdentityBuffer() ; 
    GBuffer* idb = mm->getIndicesBuffer() ; 


    optix::GeometryTriangles gtri = m_context->createGeometryTriangles();

    RTformat identityFormat = RT_FORMAT_UNSIGNED_INT4 ;  
    optix::Buffer identityBuffer = createInputBuffer<optix::uint4>( rib, identityFormat, 1, "identityBuffer"); 
    gtri["identityBuffer"]->setBuffer(identityBuffer);

    RTformat vertexFormat = RT_FORMAT_FLOAT3 ;
    optix::Buffer vertexBuffer = createInputBuffer<optix::float3>( vtx , vertexFormat, 1, "vertexBuffer"); 
    gtri["vertexBuffer"]->setBuffer(vertexBuffer);

    RTformat indexFormat = RT_FORMAT_UNSIGNED_INT3 ;        // are "coercing" from underlying int buffer 
    optix::Buffer indexBuffer = createInputBuffer<optix::uint3>( idb, indexFormat, 3 , "indexBuffer");  // need the 3 to fold for faces
    gtri["indexBuffer"]->setBuffer(indexBuffer);

    optix::Buffer emptyBuffer = m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, 0);
    gtri["tangentBuffer"]->setBuffer(emptyBuffer);
    gtri["bitangentBuffer"]->setBuffer(emptyBuffer);
    gtri["normalBuffer"]->setBuffer(emptyBuffer);
    gtri["texCoordBuffer"]->setBuffer(emptyBuffer);
 
    gtri["primitive_count"]->setUint( numFaces );  // needed for instanced offsets into buffers, so must describe the buffer, NOT the intent 
    gtri["repeat_index"]->setUint(mm->getIndex()); 

    gtri->setPrimitiveCount( numFaces );
    gtri->setTriangleIndices( indexBuffer, indexFormat );
    gtri->setVertices( numVertices, vertexBuffer, vertexFormat ); 
    gtri->setBuildFlags( RTgeometrybuildflags( 0 ) );

    optix::Program gtri_att = m_ocontext->createProgram("GeometryTriangles.cu", "triangle_attributes");
    gtri->setAttributeProgram( gtri_att );  

    return gtri  ; 
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

optix::Geometry OGeo::makeTriangulatedGeometry(GMergedMesh* mm)
{
    unsigned numVolumes = mm->getNumVolumes();
    unsigned numFaces = mm->getNumFaces();
    unsigned numITransforms = mm->getNumITransforms();

    LOG(LEVEL) 
        << " mmIndex " << mm->getIndex() 
        << " numFaces (PrimitiveCount) " << numFaces
        << " numVolumes " << numVolumes
        << " numITransforms " << numITransforms 
        ;
             
    GBuffer* id = mm->getAppropriateRepeatedIdentityBuffer();
    GBuffer* vb = mm->getVerticesBuffer() ; 
    GBuffer* ib = mm->getIndicesBuffer() ;  


    optix::Geometry geometry = m_context->createGeometry();
    geometry->setIntersectionProgram(m_ocontext->createProgram("TriangleMesh.cu", "mesh_intersect"));
    geometry->setBoundingBoxProgram(m_ocontext->createProgram("TriangleMesh.cu", "mesh_bounds"));

    optix::Buffer identityBuffer = createInputBuffer<optix::uint4>( id, RT_FORMAT_UNSIGNED_INT4, 1 , "identityBuffer"); 
    geometry["identityBuffer"]->setBuffer(identityBuffer);

    optix::Buffer vertexBuffer = createInputBuffer<optix::float3>( vb, RT_FORMAT_FLOAT3, 1, "vertexBuffer" ); 
    geometry["vertexBuffer"]->setBuffer(vertexBuffer);

    optix::Buffer indexBuffer = createInputBuffer<optix::int3>( ib, RT_FORMAT_INT3, 3 , "indexBuffer");  // need the 3 to fold for faces
    geometry["indexBuffer"]->setBuffer(indexBuffer);

    optix::Buffer emptyBuffer = m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, 0);
    geometry["tangentBuffer"]->setBuffer(emptyBuffer);
    geometry["bitangentBuffer"]->setBuffer(emptyBuffer);
    geometry["normalBuffer"]->setBuffer(emptyBuffer);
    geometry["texCoordBuffer"]->setBuffer(emptyBuffer);

    geometry->setPrimitiveCount( numFaces ); 
    geometry["primitive_count"]->setUint( numFaces ); 
    geometry["repeat_index"]->setUint( mm->getIndex() );  // non-instanced
 
    return geometry ; 
}

const char* OGeo::getContextName() const 
{
    std::stringstream ss ; 
    ss << "OGeo"
       << m_mmidx 
        ; 
      
    std::string name = ss.str();  
    return strdup(name.c_str()); 
}


/**
*reuse* was an unsuccessful former attempt to "purloin the OpenGL buffers" avoid duplicating geometry 
info between OpenGL and OptiX setting reuse to true causes OptiX/OpenGL launch failure : bad enum 
**/

template <typename T>
optix::Buffer OGeo::createInputBuffer(GBuffer* buf, RTformat format, unsigned int fold, const char* name, bool reuse)
{
   //TODO: eliminate use of this, moving to NPY buffers instead

   unsigned bytes = buf->getNumBytes() ;

   unsigned bit = buf->getNumItems() ; 
   unsigned nit = bit/fold ; 
   unsigned nel = buf->getNumElements();
   unsigned mul = OFormat::Multiplicity(format) ;

   int buffer_id = buf->getBufferId() ;


   if(m_verbosity > 2)
   LOG(info)
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
   LOG(info)
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
    LOG(info) 
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




