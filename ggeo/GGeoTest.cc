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

#include <iomanip>

#include "BBnd.hh"
#include "BStr.hh"

// npy-
#include "NSlice.hpp"
#include "NCSG.hpp"
#include "NCSGList.hpp"
#include "GLMFormat.hpp"
#include "NGLMExt.hpp"
#include "NLODConfig.hpp"
#include "NBBox.hpp"
#include "NQuad.hpp"


// okc-
#include "Opticks.hh"
#include "OpticksIdentity.hh"
#include "OpticksResource.hh"
#include "OpticksEventAna.hh"
#include "OpticksConst.hh"


#include "GVector.hh"
#include "GGeoBase.hh"
#include "GGeoLib.hh"

#include "GMaterialLib.hh"
#include "GSurfaceLib.hh"
#include "GBndLib.hh"

#include "GMeshLib.hh"

#include "GMergedMesh.hh"
#include "GVolume.hh"

#include "GNodeLib.hh"
#include "GNode.hh"

#include "GMaker.hh"
#include "GMeshMaker.hh"
#include "GItemList.hh"

#include "GParts.hh"
#include "GPts.hh"

#include "GTransforms.hh"
#include "GIds.hh"

#include "NGeoTestConfig.hpp"
#include "GGeoTest.hh"

#include "PLOG.hh"


const plog::Severity GGeoTest::LEVEL = PLOG::EnvLevel("GGeoTest", "DEBUG") ; 

const char* GGeoTest::UNIVERSE_LV = "UNIVERSE_LV" ; 
const char* GGeoTest::UNIVERSE_PV = "UNIVERSE_PV" ; 


NCSGList*       GGeoTest::getCSGList() const  {  return m_csglist ;  }
NGeoTestConfig* GGeoTest::getConfig() {          return m_config ; }
NCSG*           GGeoTest::findEmitter() const  { return m_csglist ? m_csglist->findEmitter() : NULL ; }
NCSG*           GGeoTest::getUniverse() const  { return m_csglist ? m_csglist->getUniverse() : NULL ; }


// pass along from basis
GScintillatorLib* GGeoTest::getScintillatorLib() const { return m_basis->getScintillatorLib() ; }
GSourceLib*       GGeoTest::getSourceLib() const {       return m_basis->getSourceLib() ; }



// local residents backed by corresponding basis libs 
GBndLib*          GGeoTest::getBndLib() const {          return m_bndlib ;  }
GMeshLib*         GGeoTest::getMeshLib() const {         return m_basemeshlib ;  }
GSurfaceLib*      GGeoTest::getSurfaceLib() const {      return m_slib ;  }
GMaterialLib*     GGeoTest::getMaterialLib() const {     return m_mlib ;  }

// locally customized 
const char*       GGeoTest::getIdentifier() const {       return "GGeoTest" ; }


GMergedMesh*      GGeoTest::getMergedMesh(unsigned index) const { return m_geolib->getMergedMesh(index) ; }
unsigned          GGeoTest::getNumMergedMesh() const {            return m_geolib->getNumMergedMesh() ; }


GGeoLib*          GGeoTest::getGeoLib() const {                   return m_geolib ; }
GNodeLib*         GGeoTest::getNodeLib() const {                  return m_nodelib ; }



void GGeoTest::dump(const char* msg)
{
    LOG(info) << msg  ; 
}
void GGeoTest::setErr(int err)  
{
   m_err = err ;  
}
int GGeoTest::getErr() const 
{
   return m_err  ;  
}



GGeoTest::GGeoTest(Opticks* ok, GGeoBase* basis) 
    :  
    m_ok(ok),
    m_dbggeotest(ok->isDbgGeoTest()),    // --dbggeotest
    m_config_(ok->getTestConfig()),
    m_config(new NGeoTestConfig(m_config_)),
    m_verbosity(m_ok->isDbgGeoTest() ? 10 : m_config->getVerbosity()),
    m_resource(ok->getResource()),
    m_dbgbnd(m_ok->isDbgBnd()),
    m_dbganalytic(m_ok->isDbgAnalytic()),
    m_lodconfig(ok->getLODConfig()),
    m_lod(ok->getLOD()),
    m_input_analytic(m_config->getAnalytic()),
    m_csgpath(m_config->getCSGPath()),
    m_test(true),
    m_basis(basis),
    m_basemeshlib(basis->getMeshLib()),
    m_basegeolib(basis->getGeoLib()), 
    m_mlib(new GMaterialLib(m_ok, basis->getMaterialLib())),
    m_slib(new GSurfaceLib(m_ok, basis->getSurfaceLib())),
    m_bndlib(new GBndLib(m_ok, m_mlib, m_slib)),
    m_geolib(new GGeoLib(m_ok, m_bndlib)),
    m_nodelib(new GNodeLib(m_ok)),
    m_meshlib(new GMeshLib(m_ok)),
    m_maker(new GMaker(m_ok, m_bndlib, m_basemeshlib)),
    m_csglist(m_csgpath ? NCSGList::Load(m_csgpath, m_verbosity ) : NULL),
    m_numtree( m_csglist ? m_csglist->getNumTrees() : 0 ),
    m_err(0)
{
    assert(m_basis); 
    assert(m_input_analytic) ;  // <-- test input geometry always analytic, but the output form used in OGeo could be tri  

    checkPts(); 

    init();
}

void GGeoTest::init()
{
    LOG(LEVEL) << "[" ;

    bool is_ncsg = m_config->isNCSG() ; 
    if(!is_ncsg)
    {
        LOG(fatal) << "expecting isNCSG config" ; 
        std::cout << " m_config_ " << ( m_config_ ? m_config_ : "NULL" ) << std::endl ; 
    }
    assert(is_ncsg); 

    GMergedMesh* tmm_ = initCreateCSG() ;

    if(!tmm_)
    {
        setErr(101) ; 
        return ;        
    }

    assert( m_lod == 0); 

    GMergedMesh* tmm = m_lod > 0 ? GMergedMesh::MakeLODComposite(tmm_, m_lodconfig->levels ) : tmm_ ;         


    bool output_analytic = m_ok->isXAnalytic();  

    char geocode =  output_analytic ? OpticksConst::GEOCODE_ANALYTIC : OpticksConst::GEOCODE_TRIANGULATED ;  // message to OGeo

    tmm->setGeoCode( geocode );



    m_geolib->setMergedMesh( 0, tmm );  // TODO: create via standard GGeoLib::create ?

    LOG(LEVEL) << "]" ;
}


/**
GGeoTest::checkPts
-------------------

**/

void GGeoTest::checkPts()
{
    if(m_csglist == NULL ) return ;  // OpticksHubTest
    NCSG* proxy = m_csglist->findProxy();  
    

    int proxy_idx = m_csglist->findProxyIndex(); 
    LOG(LEVEL)
        << " proxy " << proxy 
        << " proxy_idx " << proxy_idx
        ;  

    unsigned nmm = m_basegeolib->getNumMergedMesh() ; 
    LOG(LEVEL) 
        << " basegeolib " 
        << " nmm " << nmm 
        ; 

    for(unsigned i=0 ; i < nmm ; i++)
    {
        GMergedMesh* bmm = m_basegeolib->getMergedMesh(i);  
        GPts* pts = bmm->getPts(); 
        assert( pts ); 
        LOG(LEVEL) << std::setw(3) << i << " " << pts->brief() ;  
        //pts->dump("GGeoTest::checkPts") ; 
    }
}

/**
GGeoTest::initCreateCSG
-------------------------

testauto mode  TODO:review to see if still relevant
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Override commandline settings, and CSG boundaries and emit config
that is the whole point of --testauto to take control 
and simplify the repurposing of a test geometry 
for NCSGIntersect testing with seqmap asserts
 
**/


GMergedMesh* GGeoTest::initCreateCSG()
{
    assert(m_csgpath && "m_csgpath is required");
    assert(strlen(m_csgpath) > 3 && "unreasonable csgpath strlen");  

    LOG(LEVEL) << " m_csgpath " << m_csgpath ; 

    m_resource->setTestCSGPath(m_csgpath); // take note of path, for inclusion in event metadata
    m_resource->setTestConfig(m_config_); // take note of config, for inclusion in event metadata

    if(!m_csglist) return NULL ; 

    assert(m_csglist && "failed to load NCSGList");
    assert(m_input_analytic == true);
    assert(m_config->isNCSG());

    //unsigned numTree = m_csglist->getNumTrees() ;
    if(m_numtree == 0 )
    {
        LOG(error) << "failed to load trees" ; 
        return NULL ; 
    }

    if(m_ok->isTestAuto())
    {
        const char* autoseqmap = m_config->getAutoSeqMap();
        m_ok->setSeqMapString(autoseqmap);
        m_csglist->autoTestSetup(m_config);
    }

    GVolume* top = importCSG();

    assignBoundaries(top); 

    collectNodes(top); 


    unsigned ridx = 0 ; 
    GNode* base = NULL ; 
    GNode* root = top ; 

    GMergedMesh* tmm = GMergedMesh::Create(ridx, base, root );


    // below normally done in  GGeo::deferredCreateGParts  when not --test
    // TODO: arrange to use standard method for testing too 
    {
        GPts* pts = tmm->getPts();  

        const std::vector<const NCSG*>& solids = m_meshlib->getSolids(); 
  
        GParts* parts = GParts::Create( pts, solids ) ; 
        parts->setBndLib(m_bndlib); 
        parts->close(); 

        tmm->setParts(parts);  
    }


    std::vector<const GNode*> placements ; 
    placements.push_back(top); 
    tmm->addInstancedBuffers(placements); 


    glm::vec4 ce = tmm->getCE(0);    
    LOG(LEVEL) << " tmm.ce " << gformat(ce) ; 

    return tmm ; 
}



/**
Hmm how to use standand GNodeLib machinery with test geometry ?
To avoid placeholders. 
**/

unsigned GGeoTest::getNumTransforms() const 
{
    return 1 ; 
}
glm::mat4 GGeoTest::getTransform(unsigned index) const 
{
    glm::mat4 tr(1.f); 
    return tr ; 
}
glm::vec4 GGeoTest::getCE(unsigned index) const 
{    
    float extent = 1000.f ; 
    glm::vec4 ce(0.f,0.f,0.f, extent); 
    return ce ; 
}



/**
GGeoTest::importCSG
--------------------

Imports CSG trees from the m_csglist adding GVolume instances to the local GNodeLib 
assuming tree order is from outermost to innermost volume. 

1. adds test materials to m_mlib
2. reuse materials mentioned in test geometry boundary specification strings 
   by "stealing" material pointers from the basis lib and adding them to m_mlib
3. for each NCSG tree in the test geometry make a corresponding GVolume (with GMaker), 

   * parent/child links are set assuming simple Russian doll containment 
     with outermost volume being the first in the list of NCSG trees

   * surfaces referenced by boundaries of the test geometry are collected 
     into m_slib (stealing pointers) and surface metadata changed for test volume
     names

* materials and surfaces must be in place before adding 
  the boundary spec to get the boundary index 

**/

GVolume* GGeoTest::importCSG()
{
    LOG(LEVEL) << "[" ; 
    m_mlib->addTestMaterials(); 

    reuseMaterials(m_csglist);
   
    prepareMeshes();     

    adjustContainer(); 

    int primIdx(-1) ; 

    GVolume* top = NULL ; 
    GVolume* prior = NULL ; 

    unsigned num_mesh = m_meshlib->getNumMeshes(); 

    for(unsigned i=0 ; i < num_mesh ; i++)
    {
        primIdx++ ; // each tree is separate OptiX primitive, with own line in the primBuffer 

        GMesh* mesh = m_meshlib->getMeshSimple(i); 

        unsigned ndIdx = i ;  
        GVolume* volume = m_maker->makeVolumeFromMesh(ndIdx, mesh);
        if( top == NULL ) top = volume ;  

        if(prior)
        {
            volume->setParent(prior);
            prior->addChild(volume);
        }
        prior = volume ; 


        int sensorIndex = -1 ; 
        volume->setSensorIndex(sensorIndex); // see notes/issues/GGeoTest_GMergedMesh_mergeVolumeFaces_assert_sensor_indices.rst 

        // forming tripletIdentity here just to match the normal way of doing things, 
        // which is mostly redundant for non-instanced test geometry
        unsigned ridx = 0 ; // remainder volume
        unsigned pidx = 0 ; // only one placement
        unsigned oidx = i ; // offset-index 
        unsigned tripletIdentity = OpticksIdentity::Encode(ridx, pidx, oidx); 
        assert( oidx == tripletIdentity ); 

        volume->setTripletIdentity(tripletIdentity); 

        GPt* pt = volume->getPt(); 
        assert( pt );

        const NCSG* csg = mesh->getCSG(); 
        const char* spec = csg->getBoundary(); 
        assert( spec ); 
        LOG(LEVEL) << "boundary spec " << spec ; 

        relocateSurfaces(volume, spec);

        // m_nodelib->addVolume(volume); 
        // this is too soon to be collecting volumes, manifests with assert due to boundary unset.
        // Instead moved to collectNodes, see notes/issues/G4Opticks_GGeo_rejig_shakedown.rst 
    }
    LOG(LEVEL) << "]" ; 

    return top ; 
}

/**
GGeoTest::collectNodes
------------------------

Recursive traversal collecting all nodes into m_nodelib:GNodeLib.

NB Must be invoked late, as adding volumes to GNodeLib invokes GVolume::getIdentity
which asserts when things such as the boundary are unset.

Same as GInstancer::collectNodes GInstancer::collectNodes_r for normal geometry

**/

void GGeoTest::collectNodes(const GVolume* root)
{
    assert(root);
    collectNodes_r(root, 0); 
}
void GGeoTest::collectNodes_r(const GNode* node, unsigned depth )
{
    const GVolume* volume = dynamic_cast<const GVolume*>(node); 
    m_nodelib->addVolume(volume); 
    for(unsigned i = 0; i < node->getNumChildren(); i++) collectNodes_r(node->getChild(i), depth + 1 );
} 








/**
GGeoTest::prepareMeshes
------------------------------

Proxied in geometry is centered

**/

void GGeoTest::prepareMeshes()
{
    LOG(LEVEL) << "[" ; 

    assert(m_csgpath);
    assert(m_csglist);
    //unsigned numTree = m_csglist->getNumTrees() ;

    assert( m_numtree > 0 );
    for(unsigned i=0 ; i < m_numtree ; i++)
    {
        NCSG* tree = m_csglist->getTree(i) ; 
        GMesh* mesh =  tree->isProxy() ? importMeshViaProxy(tree) : m_maker->makeMeshFromCSG(tree) ; 
        const char* name = BStr::concat<unsigned>("testmesh", i, NULL ); 
        mesh->setName(name);   

        if(m_dbggeotest) 
            mesh->Summary("GGeoTest::prepareMeshes"); 

        mesh->setIndex(m_meshlib->getNumMeshes());   // <-- used for GPt reference into GMeshLib.m_meshes
        m_meshlib->add(mesh); 
    }

    LOG(LEVEL)  
        << "]"  
        << " csgpath " << m_csgpath 
        << " m_numtree " << m_numtree 
        << " verbosity " << m_verbosity
        ;
}

/**
GGeoTest::importMeshViaProxy
------------------------------

Proxy CSG solids have the proxylv attribute set to
a value greater than -1 which refers to a solid from
the basis geometry.  

GMeshLib loaded GMesh have an associated NCSG instance
with the analytic representation of the solid.

**/

GMesh* GGeoTest::importMeshViaProxy(NCSG* proxy)
{
    assert( proxy->isProxy() ); 

    // property of the proxy that refers to the original index of the proxied
    unsigned lvIdx = proxy->getProxyLV(); 

    // NB these are properties of the proxy, NOT the proxied
    const char* spec = proxy->getBoundary();  
    unsigned index = proxy->getIndex(); 

    assert( spec ); 

    LOG(LEVEL) 
        << "["
        << " proxy.index " << index
        << " proxy.spec " << spec 
        << " proxyLV " << lvIdx 
        ; 

    GMesh* mesh = m_basemeshlib->getMeshSimple(lvIdx); 
    assert( mesh ); 
    const NCSG* csg = mesh->getCSG(); 
    assert( csg ) ;

    const GMesh* altmesh = mesh->getAlt(); 

    if( altmesh )
    {
        const NCSG* altcsg = altmesh->getCSG() ; 
        LOG(LEVEL) 
            << " csg.is_balanced " << csg->is_balanced()
            << " altcsg.is_balanced " << altcsg->is_balanced()
            ; 
    }

    assert( csg->getBoundary() == NULL && "expecting fresh csg from basemeshlib to have no boundary assigned") ; 
    const_cast<NCSG*>(csg)->setOther(proxy);  // keep note of the proxy from whence it came

    mesh->setIndex( index ) ; 
    mesh->setCSGBoundary( spec );  // adopt the boundary from the proxy object setup in python

    mesh->applyCentering();        // applies to both GMesh and NCSG instances

    LOG(LEVEL) << "]" ; 
    return mesh ;  
}


/**
GGeoTest::adjustContainer
--------------------------

Changes the size of the container to fit the solid brought in by 
the proxy by inserting the replacement NCSG solid into the 
m_csglist and then invoking NCSGList::adjustContainerSize 

That updates the analytic geometry, but also need to update the
triangulated geometry GMesh for the OpenGL visualization.
So use GMaker polygonization to do so, replacinging the 
mesh with a newly created one.

This is kinda awkward because its modifying this,
 would be cleaner just to create a container GVolume/GMesh/NCSG 
combo in separate method from a bb or ce or just extent.

**/

void GGeoTest::adjustContainer()
{
    bool has_container = m_csglist->hasContainer() ; 
    if(!has_container) return ; 

    NCSG* container = m_csglist->findContainer(); 
    assert(container) ; 
    bool containerautosize = container->isContainerAutoSize() ; 

    if(!containerautosize)
    {
        LOG(LEVEL) << " containerautosize DISABLED by metadata on container CSG " << containerautosize  ;
        return ; 
    }  
    else
    {
        LOG(LEVEL) << " containerautosize ENABLED by metadata on container CSG " << containerautosize  ;
    }

    bool has_proxy = m_csglist->hasProxy() ; 
    if(has_proxy) 
    {
        updateWithProxiedSolid() ;    
    }

    LOG(LEVEL)
        << " m_numtree " << m_numtree
        ; 

    m_csglist->update();  

    // Find the adjusted analytic container (NCSG) and create corresponding 
    // triangulated geometry (GMesh) to replace the old one

    int container_index = m_csglist->findContainerIndex() ; 
    assert( container_index > -1 ) ; 

    nbbox container_bba = container->bbox(); 

    LOG(LEVEL) 
        << " container_index " << container_index
        << " container_bba " << container_bba.desc()
        ; 


    GMesh* replacement_mesh = GMeshMaker::Make(container_bba); 
    replacement_mesh->setIndex( container_index ) ; 
    replacement_mesh->setCSG(container); 

    m_meshlib->replace( container_index, replacement_mesh); 

    if(m_dbggeotest) 
    {
        LOG(info) 
            << " container_index " << container_index
            << " container_bba " << container_bba.description() 
            ;

        replacement_mesh->Summary("GGeoTest::adjustContainer.replacement_mesh"); 
    }
}



/**
GGeoTest::updateWithProxiedSolid
------------------------------------

Lookup the proxied analytic solid (NCSG) from 
basis geocache and insert it into the m_csglist 
 
The proxied mesh is already incorporated into m_meshes 
by prepareMeshes

HMM : a more "together" way of doing this would be better

**/

void GGeoTest::updateWithProxiedSolid()
{
    LOG(LEVEL) << "[" ; 

    bool has_proxy = m_csglist->hasProxy() ; 
    assert( has_proxy ); 

    NCSG* proxy = m_csglist->findProxy(); 
    assert( proxy ) ; 

    unsigned proxy_index = m_csglist->findProxyIndex();  

    unsigned num_mesh = m_meshlib->getNumMeshes(); 

    LOG(LEVEL) 
        << " proxy_index " << proxy_index
        << " num_mesh " << num_mesh 
        ;

    assert( proxy_index < num_mesh  ) ; 

    GMesh* viaproxy = m_meshlib->getMeshSimple(proxy_index) ;
 
    const NCSG* replacement_solid = viaproxy->getCSG();  
    unsigned index = viaproxy->getIndex();  
    assert( index == proxy_index ); 

    nbbox bb = replacement_solid->bbox(); 

    LOG(LEVEL) 
        << " replacement_solid "  
        << bb.desc()
        ; 

    m_csglist->setTree( index, const_cast<NCSG*>(replacement_solid) ); 
    // ^^^ awkward reachback : perhaps update this on proxying 


    LOG(LEVEL) << "]" ; 
}






/**
GGeoTest::assignBoundaries
------------------------

1. boundaries used in the test geometry are added to m_bndlib and the 
   boundary index is assigned to GVolume

Final pass setting boundaries
as all mat/sur must be added to the mlib/slib
before can form boundaries. As boundaries 
require getIndex calls that will close the slib, mlib 
(settling the indices) and making subsequnent mat/sur additions assert.

Note that late setting of boundary is fine for GParts (analytic geometry), 
as spec are held as strings within GParts until GParts::close

See notes/issues/GGeoTest_isClosed_assert.rst 

**/

void GGeoTest::assignBoundaries(GVolume* root)
{
    plog::Severity level = m_dbggeotest ? info : debug ;     

    LOG(level) << "[" ; 

    unsigned numTree = m_csglist->getNumTrees() ;

    m_bndlib->closeConstituents(); 

    assignBoundaries_r(root, 0); 

    // see notes/issues/material-names-wrong-python-side.rst
    LOG(level) << "Save mlib/slib names " 
              << " numTree : " << numTree
              << " csgpath : " << m_csgpath
              ;

    if( numTree > 0 )
    { 
        m_mlib->saveNames(m_csgpath);
        m_slib->saveNames(m_csgpath);
    } 

    LOG(level) << "]" ; 
}


void GGeoTest::assignBoundaries_r(GNode* node, unsigned depth )
{
    GVolume* volume = dynamic_cast<GVolume*>(node); 
    const NCSG* csg = volume->getMesh()->getCSG(); 
    const char* spec = csg->getBoundary();  
    assert(spec);  
    unsigned boundary = m_bndlib->addBoundary(spec, false); 
    volume->setBoundary(boundary); // creates arrays, duplicating boundary to all tris
 
    for(unsigned i = 0; i < node->getNumChildren(); i++) assignBoundaries_r(node->getChild(i), depth + 1 );
} 






NCSG* GGeoTest::getTree(unsigned index) const 
{
    return m_csglist->getTree(index);
}
unsigned GGeoTest::getNumTrees() const 
{
    return m_csglist->getNumTrees();
}


/**
GGeoTest::relocateSurfaces
----------------------------

1. if the inner and outer surfaces referenced by name in the spec string 
   are already present in m_slib then there is nothing to do 

2. otherwise relocate surfaces using volume names of the test geometry, this 
   "steals" surface pointers from the basis library.  And it modifies the
   metadata of the surfaces using test geometry names.
   
   * Hmm: so it is breaking the basis surface lib.

**/

void GGeoTest::relocateSurfaces(GVolume* volume, const char* spec)
{

    if(m_dbgbnd)
    { 
        LOG(error) << "[--dbgbnd] this.slib  " << m_slib->desc()  ; 
        LOG(error) << "[--dbgbnd] basis.slib " << m_slib->getBasis()->desc()  ; 
    }


    BBnd b(spec);
    bool unknown_osur = b.osur && !m_slib->hasSurface(b.osur) ;
    bool unknown_isur = b.isur && !m_slib->hasSurface(b.isur) ;

    if(unknown_osur || unknown_isur)
    {
        GVolume* parent = static_cast<GVolume*>(volume->getParent()) ; 
        const char* self_lv = volume->getLVName() ;
        const char* self_pv = volume->getPVName() ;   
        const char* parent_pv = parent ? parent->getPVName() : UNIVERSE_PV ; 

        if(m_dbgbnd)
        LOG(error) 
              << "[--dbgbnd]"
              << " spec " << spec
              << " unknown_osur " << unknown_osur
              << " unknown_isur " << unknown_isur
              << " self_lv " << self_lv
              << " self_pv " << self_pv
              << " parent_pv " << parent_pv
              ;

        if( b.osur && b.isur && strcmp(b.osur, b.isur) == 0 ) // skin  : same outer and inner surface name indicates skin
        {
            m_slib->relocateBasisSkinSurface( b.osur, self_lv );
        }
        else 
        {
            if(unknown_isur)  // inner border self->parent
                m_slib->relocateBasisBorderSurface( b.isur, self_pv, parent_pv  );
       
            if(unknown_osur)  // outer border parent->self
                m_slib->relocateBasisBorderSurface( b.osur, parent_pv, self_pv ) ; 
        }
    }
}


/**
GGeoTest::reuseMaterials
-------------------------

Reusing materials from the basis geometry means that 
they are added into the local GGeoTest material lib.

**/

void GGeoTest::reuseMaterials(NCSGList* csglist)
{
    // reuse all the materials first, to prevent premature GPropLib close
    unsigned num_tree = csglist->getNumTrees() ;
    for(unsigned i=0 ; i < num_tree ; i++)
    {
        NCSG* tree = csglist->getTree(i) ; 
        const char* spec = tree->getBoundary();  
        reuseMaterials(spec);
    }
}

void GGeoTest::reuseMaterials(const char* spec)
{
    BBnd b(spec);
    if(strcmp(b.omat, b.imat) == 0)     // same inner and outer material
    {
        if(!m_mlib->hasMaterial(b.omat)) m_mlib->reuseBasisMaterial( b.omat );
    } 
    else
    {
        if(!m_mlib->hasMaterial(b.omat)) m_mlib->reuseBasisMaterial( b.omat );
        if(!m_mlib->hasMaterial(b.imat)) m_mlib->reuseBasisMaterial( b.imat );
    }
}




/**
GGeoTest::MakeArgForce
------------------------

Used by OpticksHubTest 

TODO: maybe eliminate or move to test, once return to that context

**/
const char* GGeoTest::MakeArgForce(const char* funcname, const char* extra)
{
    std::string argforce = MakeArgForce_(funcname, extra);
    return strdup(argforce.c_str());
}


std::string GGeoTest::MakeArgForce_(const char* funcname, const char* extra)
{
    std::stringstream ss ; 
    ss  
       << "--test"
       << " " 
       << "--testconfig"
       << " " 
       << MakeTestConfig_(funcname)
       ;   

    if(extra) ss << " " << extra ; 
    return ss.str() ;
}

std::string GGeoTest::MakeTestConfig_(const char* funcname)
{
    std::stringstream ss ; 
    ss  
       << "analytic=1"
       << "_" 
       << "mode=PyCsgInBox"
       << "_" 
       << "outerfirst=1"
       << "_" 
       << "csgpath=$TMP/" << funcname
       << "_" 
       << "name=" << funcname
       ;   

    return ss.str() ;
}



/**
GGeoTest::anaEvent
---------------------

This is invoked by OpticksHub::anaEvent 

**/

void GGeoTest::anaEvent(OpticksEvent* evt)
{
    int dbgnode = m_ok->getDbgNode();
    //NCSG* csg = getTree(dbgnode);

    LOG(info) 
        << " dbgnode " << dbgnode
        << " numTrees " << getNumTrees()
        << " evt " << evt
        ;

    assert( m_csglist ) ;  

    OpticksEventAna ana(m_ok, evt, m_csglist);
    ana.dump("GGeoTest::anaEvent");
}


