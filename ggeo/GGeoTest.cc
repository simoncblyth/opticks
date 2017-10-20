#include <iomanip>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>

#include "BStr.hh"

// npy-
#include "NSlice.hpp"
#include "NCSG.hpp"
#include "GLMFormat.hpp"
#include "NGLMExt.hpp"
#include "NLODConfig.hpp"

// opticks-
#include "Opticks.hh"
#include "OpticksConst.hh"


#include "GVector.hh"
#include "GGeo.hh"
#include "GGeoLib.hh"
#include "GBndLib.hh"
#include "GMergedMesh.hh"
#include "GPmt.hh"
#include "GSolid.hh"
#include "GMaker.hh"
#include "GItemList.hh"
#include "GParts.hh"
#include "GTransforms.hh"
#include "GIds.hh"

#include "GGeoTestConfig.hh"
#include "GGeoTest.hh"

#include "PLOG.hh"


GGeoTest::GGeoTest(Opticks* ok, GGeoTestConfig* config, GGeo* ggeo) 
    : 
    m_ok(ok),
    m_dbganalytic(m_ok->hasOpt("dbganalytic")),
    m_lodconfig(ok->getLODConfig()),
    m_lod(ok->getLOD()),
    m_config(config),
    m_ggeo(ggeo),
    m_geolib(NULL),
    m_bndlib(NULL),
    m_maker(NULL),
    m_verbosity(0)
{
    init();
}

void GGeoTest::init()
{
    if(m_ggeo)
    {
        LOG(warning) << "GGeoTest::init booting from m_ggeo " ; 
        m_geolib = m_ggeo->getGeoLib();
        m_bndlib = m_ggeo->getBndLib();
    }
    else
    {
        LOG(warning) << "GGeoTest::init booting from m_ok cache" ; 
        bool analytic = false ; 
        m_bndlib = GBndLib::load(m_ok, true );
        m_geolib = GGeoLib::Load(m_ok, analytic, m_bndlib );
    }
    m_maker = new GMaker(m_ok);
}


void GGeoTest::dump(const char* msg)
{
    LOG(info) << msg  
              ; 
}

void GGeoTest::modifyGeometry()
{
    const char* csgpath = m_config->getCsgPath();
    bool analytic = m_config->getAnalytic(); 

    if(csgpath) assert(analytic == true);

    GMergedMesh* tmm_ = create();

    GMergedMesh* tmm = m_lod > 0 ? GMergedMesh::MakeLODComposite(tmm_, m_lodconfig->levels ) : tmm_ ;         


    char geocode =  analytic ? OpticksConst::GEOCODE_ANALYTIC : OpticksConst::GEOCODE_TRIANGULATED ;  // message to OGeo
    tmm->setGeoCode( geocode );

    if(tmm->isTriangulated()) 
    { 
        tmm->setITransformsBuffer(NULL); // avoiding FaceRepeated complications 
    } 

    //tmm->dump("GGeoTest::modifyGeometry tmm ");
    m_geolib->clear();
    m_geolib->setMergedMesh( 0, tmm );
}


GMergedMesh* GGeoTest::create()
{
    const char* csgpath = m_config->getCsgPath();
    const char* mode = m_config->getMode();
    unsigned nelem = m_config->getNumElements();

    assert( mode );
    if(nelem == 0)
    {
        LOG(fatal) << " nelem zero  " ; 
        m_config->dump("GGeoTest::create ERROR numElements==0 " ); 
    }
    assert(nelem > 0);

    LOG(info) << "GGeoTest::create START " << " mode " << mode ;

    GMergedMesh* tmm = NULL ; 
    std::vector<GSolid*> solids ; 

    if(csgpath != NULL)
    {
        assert( strlen(csgpath) > 3 && "unreasonable csgpath strlen");  
        loadCSG(csgpath, solids);
        tmm = combineSolids(solids, NULL);
    }
    else if(strcmp(mode, "BoxInBox") == 0) 
    {
        createBoxInBox(solids); 
        labelPartList(solids) ;
        tmm = combineSolids(solids, NULL);
    }
    else if( strcmp(mode, "PmtInBox") == 0)
    {
        tmm = createPmtInBox(); 
    }
    else 
    { 
        LOG(fatal) << "GGeoTest::create mode not recognized " << mode ; 
        assert(0);
    }

    assert(tmm);
    LOG(info) << "GGeoTest::create DONE " << " mode " << mode ;
    return tmm ; 
}

void GGeoTest::loadCSG(const char* csgpath, std::vector<GSolid*>& solids)
{
    int verbosity = m_config->getVerbosity();
    LOG(info) << "GGeoTest::loadCSG " 
              << " csgpath " << csgpath
              << " verbosity " << verbosity
              ; 

    std::vector<NCSG*> trees ;

    int rc = NCSG::Deserialize( csgpath, trees, verbosity );
    assert(rc == 0);

    unsigned ntree = trees.size() ;

    LOG(info) << "GGeoTest::loadCSG " << csgpath << " got " << ntree << " trees " ; 

    int primIdx(-1) ; 
    for(unsigned i=0 ; i < ntree ; i++)
    {
        primIdx++ ; // each tree is separate OptiX primitive, with own line in the primBuffer 

        NCSG* tree = trees[i] ; 

        GSolid* solid = m_maker->makeFromCSG(tree, verbosity );

        GParts* pts = solid->getParts();

        pts->setIndex(0u, i);

        if(pts->isPartList())  // not doing this for NodeTree
        {
            pts->setNodeIndexAll(primIdx ); 
        }

        pts->setBndLib(m_bndlib);

        solids.push_back(solid);
    }
}

void GGeoTest::labelPartList( std::vector<GSolid*>& solids )
{
    // PartList geometry (the precursor to proper CSG Trees, usually defined in python CSG) 
    // is implemented by allowing a single "primitive" to be composed of multiple
    // "parts", the association from part to prim being 
    // controlled via the primIdx attribute of each part.
    //
    // collected pts are converted into primitives in GParts::makePrimBuffer
  
    for(unsigned i=0 ; i < solids.size() ; i++)
    {
        GSolid* solid = solids[i];
        GParts* pts = solid->getParts();
        assert(pts);
        assert(pts->isPartList());

        OpticksCSG_t csgflag = solid->getCSGFlag(); 
        int flags = csgflag ;

        pts->setIndex(0u, i);
        pts->setNodeIndex(0u, 0 );  // seems nodeIndex not used for CSG_FLAGPARTLIST 
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

GSolid* GGeoTest::makeSolidFromConfig( unsigned i )
{
    std::string node = m_config->getNodeString(i);
    OpticksCSG_t type = m_config->getTypeCode(i);

    const char* spec = m_config->getBoundary(i);
    glm::vec4 param = m_config->getParameters(i);
    glm::mat4 trans = m_config->getTransform(i);
    unsigned boundary = m_bndlib->addBoundary(spec);

    LOG(info) << "GGeoTest::makeSolidFromConfig" 
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
    if(!oktype) LOG(fatal) << "GGeoTest::makeSolidFromConfig configured node not implemented " << node ;
    assert(oktype);

    GSolid* solid = m_maker->make(i, type, param, spec );   
    GParts* pts = solid->getParts();
    assert(pts);
    pts->setPartList(); // setting primFlag to CSG_FLAGPARTLIST
    pts->setBndLib(m_bndlib) ; 

    return solid ; 
}

void GGeoTest::createBoxInBox(std::vector<GSolid*>& solids)
{
    unsigned nelem = m_config->getNumElements();
    for(unsigned i=0 ; i < nelem ; i++)
    {
        GSolid* solid = makeSolidFromConfig(i);
        solids.push_back(solid);
    }
}


void GGeoTest::finalizeAnalytic(GParts* pts, const char* containingMaterial)
{
    const char* sensorSurface = m_ok->getSensorSurface();
    pts->setContainingMaterial(containingMaterial);    // match outer material of PMT with inner material of the box
    pts->setSensorSurface(sensorSurface) ;
    pts->setBndLib(m_bndlib) ; 
    pts->setPartList(); // setting primFlag to CSG_FLAGPARTLIST
    pts->close();      // registerBoundaries, makePrimBuffer
}


GMergedMesh* GGeoTest::createPmtInBox()
{
    assert( m_config->getNumElements() == 1 && "GGeoTest::createPmtInBox expecting single container " );

    GSolid* container = makeSolidFromConfig(0); 
    const char* spec = m_config->getBoundary(0);
    const char* container_inner_material = m_bndlib->getInnerMaterialName(spec);
    int verbosity = m_config->getVerbosity();

    GMergedMesh* mmpmt = loadPmtDirty();
    assert(mmpmt);

    unsigned pmtNumSolids = mmpmt->getNumSolids() ; 
    container->setIndex( pmtNumSolids );

    LOG(info) << "GGeoTest::createPmtInBox " 
              << " spec " << spec 
              << " container_inner_material " << container_inner_material
              << " pmtNumSolids " << pmtNumSolids
              ; 


    GMesh* mesh = const_cast<GMesh*>(container->getMesh()); // TODO: reorg to avoid 
    mesh->setIndex(1000);
    
    container->getParts()->setPrimFlag(CSG_FLAGPARTLIST);  // PmtInBox uses old partlist, not the default CSG_FLAGNODETREE
    container->getParts()->setAnalyticVersion(mmpmt->getParts()->getAnalyticVersion()); // follow the PMT version for the box

    GMergedMesh* triangulated = GMergedMesh::combine( mmpmt->getIndex(), mmpmt, container, verbosity );   
    // hmm this is putting the container at the end... does that matter ?

    //if(verbosity > 1)
    triangulated->dumpSolids("GGeoTest::createPmtInBox GMergedMesh::dumpSolids combined (triangulated) ");

    GParts* analytic = triangulated->getParts();

    finalizeAnalytic( analytic, container_inner_material );

    // needed by OGeo::makeAnalyticGeometry
    NPY<unsigned int>* idBuf = mmpmt->getAnalyticInstancedIdentityBuffer();
    NPY<float>* itransforms = mmpmt->getITransformsBuffer();

    assert(idBuf);
    assert(itransforms);

    triangulated->setAnalyticInstancedIdentityBuffer(idBuf);
    triangulated->setITransformsBuffer(itransforms);

    return triangulated ; 
}

GMergedMesh* GGeoTest::combineSolids(std::vector<GSolid*>& solids, GMergedMesh* mm0)
{
    unsigned verbosity = 3 ; 
    GMergedMesh* tri = GMergedMesh::combine( 0, mm0, solids, verbosity );

    unsigned nelem = solids.size() ; 
    GTransforms* txf = GTransforms::make(nelem); // identities
    GIds*        aii = GIds::make(nelem);        // placeholder (n,4) of zeros

    tri->setAnalyticInstancedIdentityBuffer(aii->getBuffer());  
    tri->setITransformsBuffer(txf->getBuffer());

    GParts* pts0 = solids[0]->getParts();
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
        const char* msg = "GGeoTest::combineSolids --dbganalytic" ;
        pts->Summary(msg);
        pts->dumpPrimInfo(msg); // this usually dumps nothing as solid buffer not yet created
    }
    // collected pts are converted into primitives in GParts::makePrimBuffer
    return tri ; 
}

GMergedMesh* GGeoTest::loadPmtDirty()
{
    // DIRTY ASSOCIATION BETWEEN OLD STYLE ANALYTIC GPmt AND TRIANGULATED GMergedMesh 
    //
    // GPmt 
    //    detdesc parsed analytic geometry (see pmt-ecd dd.py tree.py etc..)
    //

    const char* pmtpath = m_config->getPmtPath();
    int verbosity = m_config->getVerbosity();
    assert(pmtpath);

    GMergedMesh* mmpmt = GMergedMesh::load(pmtpath);

    if(mmpmt == NULL) LOG(fatal) << "GGeoTest::loadPmtDirty FAILED " << pmtpath ;
    assert(mmpmt);

    if(verbosity > 1) mmpmt->dumpSolids("GGeoTest::loadPmtDirty GMergedMesh::dumpSolids (before:mmpmt) ");

    GPmt* pmt = m_ggeo->getPmt();
    if(!pmt) LOG(fatal) << "GPmt pmt NULL from GGeo::getPmt " ;
    assert( pmt && "GGeoTest::loadPmtDirty GGeo::getPmt returned NULL " );

    GParts* pts = pmt->getParts();
 
    mmpmt->setParts(pts); 
    
    return mmpmt ; 
}

