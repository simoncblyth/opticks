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

// okc-
#include "Opticks.hh"
#include "OpticksResource.hh"
#include "OpticksEventAna.hh"
#include "OpticksConst.hh"


#include "GVector.hh"
#include "GGeoBase.hh"
#include "GGeoLib.hh"

#include "GMaterialLib.hh"
#include "GSurfaceLib.hh"
#include "GBndLib.hh"
#include "GPmtLib.hh"
#include "GMeshLib.hh"

#include "GMergedMesh.hh"
#include "GPmt.hh"
#include "GVolume.hh"

#include "GNodeLib.hh"

#include "GMaker.hh"
#include "GMeshMaker.hh"
#include "GItemList.hh"
#include "GParts.hh"
#include "GTransforms.hh"
#include "GIds.hh"

#include "NGeoTestConfig.hpp"
#include "GGeoTest.hh"

#include "PLOG.hh"


const plog::Severity GGeoTest::LEVEL = debug ; 

const char* GGeoTest::UNIVERSE_LV = "UNIVERSE_LV" ; 
const char* GGeoTest::UNIVERSE_PV = "UNIVERSE_PV" ; 


NCSGList*       GGeoTest::getCSGList() const  {  return m_csglist ;  }
NGeoTestConfig* GGeoTest::getConfig() {          return m_config ; }
NCSG*           GGeoTest::findEmitter() const  { return m_csglist ? m_csglist->findEmitter() : NULL ; }
NCSG*           GGeoTest::getUniverse() const  { return m_csglist ? m_csglist->getUniverse() : NULL ; }


// pass along from basis
GScintillatorLib* GGeoTest::getScintillatorLib() const { return m_basis->getScintillatorLib() ; }
GSourceLib*       GGeoTest::getSourceLib() const {       return m_basis->getSourceLib() ; }

// local copy of m_basis pointer
GPmtLib*          GGeoTest::getPmtLib() const {          return m_pmtlib ; }

// local residents backed by corresponding basis libs 
GBndLib*          GGeoTest::getBndLib() const {          return m_bndlib ;  }
GMeshLib*         GGeoTest::getMeshLib() const {         return m_meshlib ;  }
GSurfaceLib*      GGeoTest::getSurfaceLib() const {      return m_slib ;  }
GMaterialLib*     GGeoTest::getMaterialLib() const {     return m_mlib ;  }

// locally customized 
const char*       GGeoTest::getIdentifier() const {       return "GGeoTest" ; }
GMergedMesh*      GGeoTest::getMergedMesh(unsigned index) const { return m_geolib->getMergedMesh(index) ; }
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
    m_analytic(m_config->getAnalytic()),
    m_csgpath(m_config->getCSGPath()),
    m_test(true),
    m_basis(basis),
    m_pmtlib(basis->getPmtLib()),
    m_meshlib(basis->getMeshLib()),
    m_mlib(new GMaterialLib(m_ok, basis->getMaterialLib())),
    m_slib(new GSurfaceLib(m_ok, basis->getSurfaceLib())),
    m_bndlib(new GBndLib(m_ok, m_mlib, m_slib)),
    m_geolib(new GGeoLib(m_ok,m_analytic,m_bndlib)),
    m_nodelib(new GNodeLib(m_ok, m_analytic, m_test, basis->getNodeLib() )),
    m_maker(new GMaker(m_ok, m_bndlib, m_meshlib)),
    m_csglist(m_csgpath ? NCSGList::Load(m_csgpath, m_verbosity ) : NULL),
    m_err(0)
{
    assert(m_basis); 
    init();
}

void GGeoTest::init()
{
    LOG(LEVEL) << "[" ;

    GMergedMesh* tmm_ = m_config->isNCSG() ? initCreateCSG() : initCreateBIB() ;

    if(!tmm_)
    {
        setErr(101) ; 
        return ;        
    }

    GMergedMesh* tmm = m_lod > 0 ? GMergedMesh::MakeLODComposite(tmm_, m_lodconfig->levels ) : tmm_ ;         

    char geocode =  m_analytic ? OpticksConst::GEOCODE_ANALYTIC : OpticksConst::GEOCODE_TRIANGULATED ;  // message to OGeo

    assert( m_analytic ) ;  

    tmm->setGeoCode( geocode );

    if(tmm->isTriangulated()) 
    { 
        tmm->setITransformsBuffer(NULL); // avoiding FaceRepeated complications 
    } 

    m_geolib->setMergedMesh( 0, tmm );  // TODO: create via standard GGeoLib::create ?

    LOG(LEVEL) << "]" ;
}


GMergedMesh* GGeoTest::initCreateCSG()
{
    assert(m_csgpath && "m_csgpath is required");
    assert(strlen(m_csgpath) > 3 && "unreasonable csgpath strlen");  

    LOG(LEVEL) << " m_csgpath " << m_csgpath ; 

    m_resource->setTestCSGPath(m_csgpath); // take note of path, for inclusion in event metadata
    m_resource->setTestConfig(m_config_); // take note of config, for inclusion in event metadata
    
    //m_resource->setEventBase(m_csgpath);   // BResource("evtbase") yields OPTICKS_EVENT_BASE 

    if(!m_csglist) return NULL ; 

    assert(m_csglist && "failed to load NCSGList");
    assert(m_analytic == true);
    assert(m_config->isNCSG());

    unsigned numTree = m_csglist->getNumTrees() ;
    if(numTree == 0 )
    {
        LOG(error) << "failed to load trees" ; 
        return NULL ; 
    }

    if(m_ok->isTestAuto())
    {
        // override commandline settings, and CSG boundaries and emit config
        // that is the whole point of --testauto to take control 
        // and simplify the repurposing of a test geometry 
        // for NCSGIntersect testing with seqmap asserts
     
        const char* autoseqmap = m_config->getAutoSeqMap();
        m_ok->setSeqMapString(autoseqmap);
        m_csglist->autoTestSetup(m_config);
    }

    importCSG();

    assignBoundaries(); 

    GMergedMesh* tmm = combineVolumes( NULL );

    return tmm ; 
}


/**
GGeoTest::importCSG
--------------------

Imports CSG trees from the m_csglist adding GVolume instances to the local GNodeLib 
assuming tree order from outermost to innermost volume. 

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


void GGeoTest::importCSG()
{
    LOG(LEVEL) << "[" ; 
    m_mlib->addTestMaterials(); 

    reuseMaterials(m_csglist);
   
    prepareMeshes(); 

    adjustContainer(); 

    int primIdx(-1) ; 

    GVolume* prior = NULL ; 

    for(unsigned i=0 ; i < m_meshes.size() ; i++)
    {
        primIdx++ ; // each tree is separate OptiX primitive, with own line in the primBuffer 

        GMesh* mesh = m_meshes[i] ; 
        GVolume* volume = m_maker->makeFromMesh(mesh);

        if(prior)
        {
            volume->setParent(prior);
            prior->addChild(volume);
        }
        prior = volume ; 

        const char* spec = mesh->getCSG()->getBoundary(); 
        assert( spec ); 

        relocateSurfaces(volume, spec);

        GParts* pts = volume->getParts();
        pts->setIndex(0u, i);
        if(pts->isPartList())  // not doing this for NodeTree
        {
            pts->setNodeIndexAll(primIdx ); 
        }
        pts->setBndLib(m_bndlib);

        m_nodelib->add(volume);
    }
    LOG(LEVEL) << "]" ; 
}





/**
GGeoTest::prepareMeshes
------------------------------

**/

void GGeoTest::prepareMeshes()
{
    assert(m_csgpath);
    assert(m_csglist);
    unsigned numTree = m_csglist->getNumTrees() ;

    assert( numTree > 0 );
    for(unsigned i=0 ; i < numTree ; i++)
    {
        NCSG* tree = m_csglist->getTree(i) ; 
        GMesh* mesh =  tree->isProxy() ? importMeshViaProxy(tree) : m_maker->makeMeshFromCSG(tree) ; 

        if(m_dbggeotest) 
            mesh->Summary("GGeoTest::prepareMeshes"); 

        m_meshes.push_back(mesh); 
    }

    if(m_dbggeotest) 
        LOG(info)  
            << " csgpath " << m_csgpath 
            << " numTree " << numTree 
            << " verbosity " << m_verbosity
            << " numMesh " << m_meshes.size()
            ;
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

**/

void GGeoTest::adjustContainer()
{
    if(!( m_csglist->hasProxy() && m_csglist->hasContainer())) 
    {
        return ;  
    }



    NCSG* container = m_csglist->findContainer(); 
    assert(container) ; 
    bool containerautosize = container->isContainerAutoSize() ; 

    if(!containerautosize)
    {
        LOG(fatal) << " containerautosize DISABLED by metadata on container CSG " << containerautosize  ;
        return ; 
    }  
    else
    {
        LOG(fatal) << " containerautosize ENABLED by metadata on container CSG " << containerautosize  ;
    }


    // Lookup the proxied analytic solid (NCSG) from 
    // basis geocache and insert it into the m_csglist 
    // then update the container size to fit it 
    //
    // The proxied mesh is already incorporated into m_meshes by prepareMeshes

    NCSG* proxy = m_csglist->findProxy(); 
    assert( proxy ) ; 


    unsigned proxy_index = m_csglist->findProxyIndex();  
    assert( proxy_index < m_meshes.size() ) ; 

    GMesh* viaproxy = m_meshes[proxy_index] ; 
    const NCSG* replacement_solid = viaproxy->getCSG();  
    unsigned index = viaproxy->getIndex();  
    assert( index == proxy_index ); 

    m_csglist->setTree( index, const_cast<NCSG*>(replacement_solid) ); 
    m_csglist->adjustContainerSize();  


    // Find the adjusted analytic container (NCSG) and create corresponding 
    // triangulated geometry (GMesh) to replace the old one


    int container_index = m_csglist->findContainerIndex() ; 
    assert( container_index > -1 ) ; 

    nbbox container_bba = container->bbox_analytic(); 

    GMesh* replacement_mesh = GMeshMaker::Make(container_bba); 
    replacement_mesh->setIndex( container_index ) ; 
    replacement_mesh->setCSG(container); 

    m_meshes[container_index] = replacement_mesh ; 

    if(m_dbggeotest) 
    {
        LOG(info) 
            << " proxy_index " << proxy_index
            << " container_index " << container_index
            << " container_bba " << container_bba.description() 
            ;

        replacement_mesh->Summary("GGeoTest::adjustContainer.replacement_mesh"); 
    }
}


/**
GGeoTest::importMeshViaProxy
------------------------------

Proxy CSG solids have the proxylv attribute set to
a value greater than -1 which refers to a solid from
the basis geometry.

**/

GMesh* GGeoTest::importMeshViaProxy(NCSG* tree)
{
    assert( tree->isProxy() ); 
    unsigned lvIdx = tree->getProxyLV(); 
    const char* spec = tree->getBoundary();  
    assert( spec ); 

    if(m_dbggeotest) 
        LOG(info) 
            << "["
            << " proxyLV " << lvIdx 
            << " proxy.spec " << spec 
            ; 

    GMesh* mesh = m_meshlib->getMeshSimple(lvIdx); 
    assert( mesh ); 
    mesh->setIndex( tree->getIndex() ) ; 

    const NCSG* csg = mesh->getCSG(); 
    assert( csg ) ; 
    assert( csg->getBoundary() == NULL && "expecting fresh csg from meshlib to have no boundary assigned") ; 

    mesh->setCSGBoundary( spec );  // adopt the boundary from the proxy object setup in python

    return mesh ;  
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

void GGeoTest::assignBoundaries()
{
    plog::Severity level = m_dbggeotest ? info : debug ;     

    LOG(level) << "[" ; 

    unsigned numTree = m_csglist->getNumTrees() ;
    unsigned numVolume = m_nodelib->getNumVolumes();
    assert( numVolume == numTree );

    m_bndlib->closeConstituents(); 

    for(unsigned i=0 ; i < numVolume ; i++)
    {
        GVolume* volume = m_nodelib->getVolume(i) ;
        const NCSG* csg = volume->getMesh()->getCSG(); 

        const char* spec = csg->getBoundary();  
        assert(spec);  
        unsigned boundary = m_bndlib->addBoundary(spec, false); 

        volume->setBoundary(boundary); // creates arrays, duplicating boundary to all tris
    }

    // see notes/issues/material-names-wrong-python-side.rst
    LOG(level) << "Save mlib/slib names " 
              << " numVolume : " << numVolume
              << " csgpath : " << m_csgpath
              ;

    if( numVolume > 0 )
    { 
        m_mlib->saveNames(m_csgpath);
        m_slib->saveNames(m_csgpath);
    } 

    LOG(level) << "]" ; 
}



/**
GGeoTest::initCreateBIB
------------------------

This is almost ready to be deleted, once 
succeed to wheel in standard solids
via some proxying metadata in the csglist.

**/

GMergedMesh* GGeoTest::initCreateBIB()
{
    const char* mode = m_config->getMode();
    unsigned nelem = m_config->getNumElements();
    if(nelem == 0 )
    {
        LOG(fatal) << " NULL csgpath and config nelem zero  " ; 
        m_config->dump("GGeoTest::initCreateBIB ERROR nelem==0 " ); 
    }
    assert(nelem > 0);

    GMergedMesh* tmm = NULL ;

    if(m_config->isBoxInBox()) 
    {
        createBoxInBox(); 
        labelPartList() ;
        tmm = combineVolumes(NULL);
    }
    else if(m_config->isPmtInBox())
    {
        assert(0); 
    }
    else 
    { 
        LOG(fatal) << "mode not recognized [" << mode << "]" ; 
        assert(0);
    }

    return tmm ; 
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
GGeoTest::labelPartList
---------------------------

PartList geometry is implemented by allowing a single "primitive" to be composed of multiple
"parts", the association from part to prim being 
controlled via the primIdx attribute of each part.

collected pts are converted into primitives in GParts::makePrimBuffer

NB Partlist test geometry was created 
   via simple simple commandline strings. 
   It is the precursor to proper CSG Trees, which are implemented 
   with NCSG and created using python opticks.analytic.csg.CSG.

**/

void GGeoTest::labelPartList()
{
    for(unsigned i=0 ; i < m_nodelib->getNumVolumes() ; i++)
    {
        GVolume* volume = m_nodelib->getVolume(i) ;
        GParts* pts = volume->getParts();
        assert(pts);
        assert(pts->isPartList());

        OpticksCSG_t csgflag = volume->getCSGFlag(); 
        int flags = csgflag ;

        pts->setIndex(0u, i);
        pts->setNodeIndex(0u, 0 );  
        //
        // for CSG_FLAGPARTLIST the nodeIndex is crucially used to associate parts to their prim 
        // setting all to zero is structuring all parts into a single prim ... 
        // can get away with that for BoxInBox (for now)
        // but would definitely not work for PmtInBox 
        //

        pts->setTypeCode(0u, flags);

        pts->setBndLib(m_bndlib);

        LOG(info) << "GGeoTest::labelPartList"
                  << " i " << std::setw(3) << i 
                  << " csgflag " << std::setw(5) << csgflag 
                  << std::setw(20) << CSGName(csgflag)
                  << " pts " << pts 
                  ;
    }
}

/**
GGeoTest::makeVolumeFromConfig
---------------------------------

* partlist geometry from commandline strings

**/

GVolume* GGeoTest::makeVolumeFromConfig( unsigned i ) // setup nodeIndex here ?
{
    std::string node = m_config->getNodeString(i);
    OpticksCSG_t type = m_config->getTypeCode(i);

    const char* spec = m_config->getBoundary(i);
    glm::vec4 param = m_config->getParameters(i);
    glm::mat4 trans = m_config->getTransform(i);
    unsigned boundary = m_bndlib->addBoundary(spec);

    LOG(info) 
        << " i " << std::setw(2) << i 
        << " node " << std::setw(20) << node
        << " type " << std::setw(2) << type 
        << " csgName " << std::setw(15) << CSGName(type)
        << " spec " << spec
        << " boundary " << boundary
        << " param " << gformat(param)
        << " trans " << gformat(trans)
        ;

    bool oktype = type < CSG_UNDEFINED ;  
    if(!oktype) LOG(fatal) << "GGeoTest::makeVolumeFromConfig configured node not implemented " << node ;
    assert(oktype);

    GVolume* volume = m_maker->make(i, type, param, spec );   
    GParts* pts = volume->getParts();
    assert(pts);
    pts->setPartList(); // setting primFlag to CSG_FLAGPARTLIST
    pts->setBndLib(m_bndlib) ; 

    return volume ; 
}


void GGeoTest::createBoxInBox()
{
    unsigned nelem = m_config->getNumElements();
    for(unsigned i=0 ; i < nelem ; i++)
    {
        GVolume* volume = makeVolumeFromConfig(i);
        m_nodelib->add(volume);
    }
}



/**
GGeoTest::createPmtInBox
--------------------------

* hmm : suspect this was a dirty hack

**/

GMergedMesh* GGeoTest::createPmtInBox()
{
    assert( m_config->getNumElements() == 1 && "GGeoTest::createPmtInBox expecting single container " );

    GVolume* container = makeVolumeFromConfig(0); 
    const char* spec = m_config->getBoundary(0);
    const char* container_inner_material = m_bndlib->getInnerMaterialName(spec);
    const char* medium = m_ok->getAnalyticPMTMedium();
    assert( strcmp( container_inner_material, medium ) == 0 );


    //GMergedMesh* mmpmt = loadPmtDirty();
    GMergedMesh* mmpmt = m_pmtlib->getPmt() ;
    assert(mmpmt);

    unsigned pmtNumVolumes = mmpmt->getNumVolumes() ; 
    container->setIndex( pmtNumVolumes );   // <-- HMM: MAYBE THIS SHOULD FEED INTO GParts::setNodeIndex ?

    LOG(info) 
        << " spec " << spec 
        << " container_inner_material " << container_inner_material
        << " pmtNumVolumes " << pmtNumVolumes
        ; 

    GMesh* mesh = const_cast<GMesh*>(container->getMesh()); // TODO: reorg to avoid 
    mesh->setIndex(1000);
    
    GParts* cpts = container->getParts() ;

    cpts->setPrimFlag(CSG_FLAGPARTLIST);  // PmtInBox uses old partlist, not the default CSG_FLAGNODETREE
    cpts->setAnalyticVersion(mmpmt->getParts()->getAnalyticVersion()); // follow the PMT version for the box
    cpts->setNodeIndex(0, pmtNumVolumes);   // NodeIndex used to associate parts to their prim, fixed 5-4-2-1-1 issue yielding 4-4-2-1-1-1

    GMergedMesh* triangulated = GMergedMesh::combine( mmpmt->getIndex(), mmpmt, container, m_verbosity );   

    // hmm this is putting the container at the end... does that matter ?

    if(m_verbosity > 1)
    triangulated->dumpVolumes("GGeoTest::createPmtInBox GMergedMesh::dumpVolumes combined (triangulated) ");

    // needed by OGeo::makeAnalyticGeometry
    NPY<unsigned int>* idBuf = mmpmt->getAnalyticInstancedIdentityBuffer();
    NPY<float>* itransforms = mmpmt->getITransformsBuffer();

    assert(idBuf);
    assert(itransforms);

    triangulated->setAnalyticInstancedIdentityBuffer(idBuf);
    triangulated->setITransformsBuffer(itransforms);

    return triangulated ; 
}


/**
GGeoTest::combineVolumes
==========================

TODO: eliminate, instead use GNodeLib::createMergeMesh 

**/

GMergedMesh* GGeoTest::combineVolumes(GMergedMesh* mm0)
{
    std::vector<GVolume*>& volumes = m_nodelib->getVolumes(); 

    LOG(LEVEL) << "[" ; 

    GMergedMesh* tri = GMergedMesh::combine( 0, mm0, volumes, m_verbosity );

    unsigned nelem = volumes.size() ; 
    GTransforms* txf = GTransforms::make(nelem); // identities
    GIds*        aii = GIds::make(nelem);        // placeholder (n,4) of zeros

    tri->setAnalyticInstancedIdentityBuffer(aii->getBuffer());  
    tri->setITransformsBuffer(txf->getBuffer());

    GParts* pts0 = volumes[0]->getParts();
    GParts* pts = tri->getParts();

    if(pts0->isPartList())
    {
        pts->setPartList();  // not too late, needed only for primBuffer creation which happens last 
    } 

    //  OGeo::makeAnalyticGeometry  requires AII and IT buffers to have same item counts

    if(m_dbganalytic)
    {
        GParts* pts = tri->getParts();
        pts->setName(m_config->getName());
        const char* msg = "GGeoTest::combineVolumes --dbganalytic" ;
        pts->Summary(msg);
        pts->dumpPrimInfo(msg); // this usually dumps nothing as solid buffer not yet created
    }
    // collected pts are converted into primitives in GParts::makePrimBuffer

    LOG(LEVEL) << "]" ; 
    return tri ; 
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

const char* GGeoTest::MakeArgForce(const char* funcname, const char* extra)
{
    std::string argforce = MakeArgForce_(funcname, extra);
    return strdup(argforce.c_str());
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


