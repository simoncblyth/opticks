#include <iomanip>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>

#include "BStr.hh"

// npy-
#include "NSlice.hpp"
#include "NCSG.hpp"
#include "GLMFormat.hpp"
#include "NGLMExt.hpp"

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


GGeoTest::GGeoTest(Opticks* opticks, GGeoTestConfig* config, GGeo* ggeo) 
    : 
    m_opticks(opticks),
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
        LOG(warning) << "GGeoTest::init booting from m_opticks cache" ; 
        bool analytic = false ; 
        m_geolib = GGeoLib::Load(m_opticks, analytic);
        m_bndlib = GBndLib::load(m_opticks, true );
    }
    m_maker = new GMaker(m_opticks);
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

    GMergedMesh* tmm = create();

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
    //TODO: unify all these modes into CSG 
    //      whilst still supporting the old partlist approach 

    const char* csgpath = m_config->getCsgPath();
    const char* mode = m_config->getMode();

    GMergedMesh* tmm = NULL ; 

    if( mode != NULL && strcmp(mode, "PmtInBox") == 0)
    {
        tmm = createPmtInBox(); 
    }
    else
    {
        std::vector<GSolid*> solids ; 
        if(csgpath != NULL)
        {
            assert( strlen(csgpath) > 3 && "unreasonable csgpath strlen");  
            loadCSG(csgpath, solids);
        }
        else
        {
            unsigned int nelem = m_config->getNumElements();
            assert(nelem > 0);
            if(     strcmp(mode, "BoxInBox") == 0) createBoxInBox(solids); 
            else  LOG(warning) << "GGeoTest::create mode not recognized " << mode ; 
        }
        tmm = combineSolids(solids);
    }
    assert(tmm);
    return tmm ; 
}

GMergedMesh* GGeoTest::createPmtInBox()
{
    // somewhat dirtily associates analytic geometry with triangulated for the PMT 
    //
    //   * detdesc parsed analytic geometry in GPmt (see pmt-ecd dd.py tree.py etc..)
    //   * instance-1 GMergedMesh 
    //
    // using prior DYB specific knowledge...
    // mergedMesh-repeat-candidate-1 is the triangulated PMT 5-solids 
    //
    // assumes single container 

    OpticksCSG_t type = m_config->getTypeCode(0);

    const char* spec = m_config->getBoundary(0);
    glm::vec4 param = m_config->getParameters(0);
    const char* container_inner_material = m_bndlib->getInnerMaterialName(spec);

    int verbosity = m_config->getVerbosity();

    LOG(info) << "GGeoTest::createPmtInBox " 
              << " type " << type
              << " csgName " << CSGName(type)
              << " spec " << spec 
              << " container_inner_material " << container_inner_material
              << " param " << gformat(param) 
              ; 

    GMergedMesh* mmpmt = loadPmt();
    unsigned int index = mmpmt->getNumSolids() ;


    GSolid* container = m_maker->make( index, type, param, spec) ;  // container box

    GMesh* mesh = const_cast<GMesh*>(container->getMesh()); // TODO: reorg to avoid 
    mesh->setIndex(1000);

    container->getParts()->setPrimFlag(CSG_FLAGPARTLIST);  // PmtInBox uses old partlist, not the default CSG_FLAGNODETREE
    container->getParts()->setAnalyticVersion(mmpmt->getParts()->getAnalyticVersion()); // follow the PMT version for the box


    // THIS IS THE DIRTY ASPECT
    //
    //    COMBINING A LOADED PMT MESH THAT JUST HAPPENS TO CORRESPOND 
    //    WITH THE ANALYTIC PARTLIST GEOMETRY
    //
    //    TODO: Use NPolygonizer to generate the mesh from the partlist ??
    //
    //

    GMergedMesh* triangulated = GMergedMesh::combine( mmpmt->getIndex(), mmpmt, container, verbosity );   

    if(verbosity > 1)
        triangulated->dumpSolids("GGeoTest::createPmtInBox GMergedMesh::dumpSolids combined (triangulated) ");

    GParts* analytic = triangulated->getParts();

    analytic->setBndLib(m_bndlib) ; // GUESS-WORK TO FIX FAILING CTestDetectorTest


    analytic->setContainingMaterial(container_inner_material);    // match outer material of PMT with inner material of the box
    analytic->setSensorSurface("lvPmtHemiCathodeSensorSurface") ; // kludge, TODO: investigate where triangulated gets this from
    analytic->close();

    // needed by OGeo::makeAnalyticGeometry

    NPY<unsigned int>* idBuf = mmpmt->getAnalyticInstancedIdentityBuffer();
    NPY<float>* itransforms = mmpmt->getITransformsBuffer();

    assert(idBuf);
    assert(itransforms);

    triangulated->setAnalyticInstancedIdentityBuffer(idBuf);
    triangulated->setITransformsBuffer(itransforms);

    return triangulated ; 
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


void GGeoTest::createBoxInBox(std::vector<GSolid*>& solids)
{
    unsigned int nelem = m_config->getNumElements();

    for(unsigned int i=0 ; i < nelem ; i++)
    {
        std::string node = m_config->getNodeString(i);
        OpticksCSG_t type = m_config->getTypeCode(i);

        const char* spec = m_config->getBoundary(i);
        glm::vec4 param = m_config->getParameters(i);
        glm::mat4 trans = m_config->getTransform(i);
        unsigned int boundary = m_bndlib->addBoundary(spec);

        LOG(info) << "GGeoTest::createBoxInBox" 
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
        if(!oktype) LOG(fatal) << "GGeoTest::createBoxInBox configured node not implemented " << node ;
        assert(oktype);

        GSolid* solid = m_maker->make(i, type, param, spec );   
        solids.push_back(solid);
    }

    int primIdx(-1) ; 

    // Boolean geometry (precursor to proper CSG Trees) 
    // is implemented by allowing 
    // a single "primitive" to be composed of multiple
    // "parts", the association from part to prim being 
    // controlled via the primIdx attribute of each part.
    //
    // collected pts are converted into primitives in GParts::makePrimBuffer
    //

    for(unsigned int i=0 ; i < solids.size() ; i++)
    {
        GSolid* solid = solids[i];
        GParts* pts = solid->getParts();
        assert(pts);
        assert(pts->isPartList());


        OpticksCSG_t csgflag = solid->getCSGFlag(); 
        int flags = csgflag ;
        if(flags == CSG_PARTLIST) primIdx++ ;   // constituents dont merit new primIdx

        pts->setIndex(0u, i);
        pts->setNodeIndex(0u, primIdx ); 
        pts->setTypeCode(0u, flags);

        pts->setBndLib(m_bndlib);

        LOG(info) << "GGeoTest::createBoxInBox"
                  << " i " << std::setw(3) << i 
                  << " primIdx " << std::setw(3) << primIdx
                  << " csgflag " << std::setw(5) << csgflag 
                  << std::setw(20) << CSGName(csgflag)
                  << " pts " << pts 
                  ;
    }
}

GMergedMesh* GGeoTest::combineSolids(std::vector<GSolid*>& solids)
{
    unsigned verbosity = 3 ; 
    GMergedMesh* tri = GMergedMesh::combine( 0, NULL, solids, verbosity );

    unsigned nelem = solids.size() ; 
    GTransforms* txf = GTransforms::make(nelem); // identities
    GIds*        aii = GIds::make(nelem);        // placeholder (n,4) of zeros

    tri->setAnalyticInstancedIdentityBuffer(aii->getBuffer());  
    tri->setITransformsBuffer(txf->getBuffer());

    //  OGeo::makeAnalyticGeometry  requires AII and IT buffers to have same item counts

    if(m_opticks->hasOpt("dbganalytic"))
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

GMergedMesh* GGeoTest::loadPmt()
{
    const char* pmtpath = m_config->getPmtPath();
    int verbosity = m_config->getVerbosity();

    GMergedMesh* mmpmt = pmtpath == NULL ? m_geolib->getMergedMesh(1) : GMergedMesh::load(pmtpath);

    if(mmpmt == NULL) LOG(fatal) << "GGeoTest::loadPmt FAILED " << pmtpath ;
    assert(mmpmt);

    if(verbosity > 1) mmpmt->dumpSolids("GGeoTest::loadPmt GMergedMesh::dumpSolids (before:mmpmt) ");

    GPmt* pmt = m_ggeo->getPmt();
    mmpmt->setParts(pmt->getParts()); // associating the analytic GPmt with the triangulated GMergedMesh 

    return mmpmt ; 
}

