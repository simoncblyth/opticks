#include <iomanip>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>



#include "BStr.hh"

// npy-
#include "NSlice.hpp"
#include "GLMFormat.hpp"

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
        m_geolib = GGeoLib::load(m_opticks);
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
    unsigned int nelem = m_config->getNumElements();
    assert(nelem > 0);

    bool analytic = m_config->getAnalytic();

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
    const char* mode = m_config->getMode();
    GMergedMesh* tmm(NULL);
 
    if(     strcmp(mode, "PmtInBox") == 0) tmm = createPmtInBox(); 
    else if(strcmp(mode, "BoxInBox") == 0) tmm = createBoxInBox(); 
    else  LOG(warning) << "GGeoTest::create mode not recognized " << mode ; 
    assert(tmm);

    return tmm ; 
}

GMergedMesh* GGeoTest::loadPmt()
{
    GMergedMesh* mmpmt = NULL ; 

    const char* pmtpath = m_config->getPmtPath();
    int verbosity = m_config->getVerbosity();

    if(pmtpath == NULL)
    {
        LOG(info) << "GGeoTest::loadPmt"
                  << " hijacking geolib mesh-1 (assumed to be PMT) "
                  ;

        mmpmt = m_geolib->getMergedMesh(1);  
    }
    else
    {
        LOG(info) << "GGeoTest::loadPmt"
                  << " from mesh with pmtpath "
                  << pmtpath
                  ;
        mmpmt = GMergedMesh::load(pmtpath);  
    }

    if(mmpmt == NULL)
        LOG(fatal) << "GGeoTest::loadPmt"
                   << " FAILED TO LOAD TESSELATED PMT FROM "
                   << pmtpath 
                   ; 
    assert(mmpmt);


    if(verbosity > 1)
    {
        LOG(info) << "GGeoTest::createPmtInBox"
                  << " verbosity " << verbosity 
                  << " numSolids " << mmpmt->getNumSolids()
                  ;

        mmpmt->dumpSolids("GGeoTest::createPmtInBox GMergedMesh::dumpSolids (before:mmpmt) ");
    }


    NSlice* slice = m_config->getSlice();

    GPmt* pmt = slice == NULL ? 
                                 m_ggeo->getPmt() 
                              : 
                                 GPmt::load( m_opticks, m_bndlib, 0, slice )  // pmtIndex:0
                              ;  


    // associating the analytic GPmt with the triangulated GMergedMesh 
    mmpmt->setParts(pmt->getParts());               

    return mmpmt ; 
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

    char nodecode = m_config->getNode(0) ;
    const char* spec = m_config->getBoundary(0);
    glm::vec4 param = m_config->getParameters(0);
    const char* container_inner_material = m_bndlib->getInnerMaterialName(spec);

    int verbosity = m_config->getVerbosity();

    LOG(info) << "GGeoTest::createPmtInBox " 
              << " nodecode " << nodecode 
              << " spec " << spec 
              << " container_inner_material " << container_inner_material
              << " param " << gformat(param) 
              ; 

    GMergedMesh* mmpmt = loadPmt();
    unsigned int index = mmpmt->getNumSolids() ;

    GSolid* solid = m_maker->make( index, nodecode, param, spec) ;
    solid->getMesh()->setIndex(1000);

    GMergedMesh* triangulated = GMergedMesh::combine( mmpmt->getIndex(), mmpmt, solid );   

    if(verbosity > 1)
        triangulated->dumpSolids("GGeoTest::createPmtInBox GMergedMesh::dumpSolids combined (triangulated) ");


    GParts* analytic = triangulated->getParts();
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






GMergedMesh* GGeoTest::createBoxInBox()
{
    std::vector<GSolid*> solids ; 
    unsigned int n = m_config->getNumElements();

    for(unsigned int i=0 ; i < n ; i++)
    {
        std::string node = m_config->getNodeString(i);
        char nodecode = m_config->getNode(i) ;
        const char* spec = m_config->getBoundary(i);
        glm::vec4 param = m_config->getParameters(i);
        glm::mat4 trans = m_config->getTransform(i);
        unsigned int boundary = m_bndlib->addBoundary(spec);

        LOG(info) << "GGeoTest::createBoxInBox" 
                  << " i " << std::setw(2) << i 
                  << " node " << std::setw(20) << node
                  << " nodecode " << std::setw(2) << nodecode 
                  << " nodename " << std::setw(15) << GMaker::NodeName(nodecode)
                  << " spec " << spec
                  << " boundary " << boundary
                  << " param " << gformat(param)
                  << " trans " << gformat(trans)
                  ;

        if(nodecode == 'U') LOG(fatal) << "GGeoTest::createBoxInBox configured node not implemented " << node ;
        assert(nodecode != 'U');

        GSolid* solid = m_maker->make(i, nodecode, param, spec );   
        solids.push_back(solid);

        // TODO: handle csg tree nodes, that break the 1-to-1 
    }


    int boolean_start = -1 ;  
    OpticksShape_t boolean_shapeflag = SHAPE_UNDEFINED ; 

    int primIdx(-1) ; 

    // Boolean geometry is implemented by allowing 
    // a single "primitive" to be composed of multiple
    // "parts", the association from part to prim being 
    // controlled via the primIdx attribute of each part.

    for(unsigned int i=0 ; i < solids.size() ; i++)
    {
        GSolid* solid = solids[i];
        GParts* pts = solid->getParts();
        assert(pts);
        OpticksShape_t shapeflag = solid->getShapeFlag(); 

        if(shapeflag == SHAPE_INTERSECTION || shapeflag == SHAPE_UNION || shapeflag == SHAPE_DIFFERENCE)
        {
            boolean_start = i ;       
            boolean_shapeflag = shapeflag ; 
        }
        int boolean_index = boolean_start > -1 ? i - boolean_start : -1 ;  

        int flags(0);
        switch(boolean_index)
        {
            case  0: flags = boolean_shapeflag | SHAPE_BOOLEAN                           ; break ; 
            case  1: flags = boolean_shapeflag | SHAPE_CONSTITUENT | SHAPE_CONSTITUENT_A ; break ; 
            case  2: flags = boolean_shapeflag | SHAPE_CONSTITUENT | SHAPE_CONSTITUENT_B ; break ; 
            default: flags = 0                                                           ; break ; 
        }

        if((flags & SHAPE_CONSTITUENT) == 0) primIdx++ ;   // constituents dont merit new primIdx

        pts->setIndex(0u, i);
        pts->setNodeIndex(0u, primIdx ); 
        pts->setFlags(0u, flags);
        pts->setBndLib(m_bndlib);

        LOG(info) << "GGeoTest::createBoxInBox"
                  << " i " << std::setw(3) << i 
                  << " primIdx " << std::setw(3) << primIdx
                  << " shapeflag " << std::setw(5) << shapeflag 
                  << std::setw(20) << ShapeName(shapeflag)
                  << " pts " << pts 
                  ;

    }

    // collected pts are converted into primitives in GParts::makePrimBuffer


    GMergedMesh* tri = GMergedMesh::combine( 0, NULL, solids );

    GTransforms* txf = GTransforms::make(n); // identities
    GIds*        aii = GIds::make(n);        // placeholder (n,4) of zeros


    tri->setAnalyticInstancedIdentityBuffer(aii->getBuffer());  
    tri->setITransformsBuffer(txf->getBuffer());

    //  OGeo::makeAnalyticGeometry  requires AII and IT buffers to have same item counts

    if(m_opticks->hasOpt("dbganalytic"))
    {
        GParts* pts = tri->getParts();
        const char* msg = "GGeoTest::createBoxInBox --dbganalytic" ;
        pts->Summary(msg);
        pts->dumpPrimInfo(msg); // this usually dumps nothing as solid buffer not yet created
    }

    return tri ; 
} 





