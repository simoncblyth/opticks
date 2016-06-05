#include "GGeoTest.hh"
#include "GGeoTestConfig.hh"

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


// npy-
#include "NLog.hpp"
#include "NSlice.hpp"
#include "GLMFormat.hpp"
#include "stringutil.hpp"

// opticks-
#include "Opticks.hh"


#include <iomanip>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>

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

    tmm->setGeoCode( analytic ? Opticks::GEOCODE_ANALYTIC : Opticks::GEOCODE_TRIANGULATED );  // to OGeo
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
                  << " from mesh "
                  << pmtpath
                  ;
        mmpmt = GMergedMesh::load(pmtpath);  
    }

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

    char shapecode = m_config->getShape(0) ;
    const char* spec = m_config->getBoundary(0);
    glm::vec4 param = m_config->getParameters(0);
    const char* container_inner_material = m_bndlib->getInnerMaterialName(spec);

    int verbosity = m_config->getVerbosity();

    LOG(info) << "GGeoTest::createPmtInBox " 
              << " shapecode " << shapecode 
              << " spec " << spec 
              << " container_inner_material " << container_inner_material
              << " param " << gformat(param) 
              ; 

    GMergedMesh* mmpmt = loadPmt();
    unsigned int index = mmpmt->getNumSolids() ;

    std::vector<GSolid*> solids = m_maker->make( index, shapecode, param, spec) ;
    for(unsigned int j=0 ; j < solids.size() ; j++)
    {
        GSolid* solid = solids[j];
        solid->getMesh()->setIndex(1000);
    }

    GMergedMesh* triangulated = GMergedMesh::combine( mmpmt->getIndex(), mmpmt, solids );   

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
        char shapecode = m_config->getShape(i) ;
        const char* spec = m_config->getBoundary(i);
        glm::vec4 param = m_config->getParameters(i);
        unsigned int boundary = m_bndlib->addBoundary(spec);

        LOG(debug) << "GGeoTest::createBoxInBox" 
                  << " i " << std::setw(2) << i 
                  << " shapecode " << std::setw(2) << shapecode 
                  << " shapename " << std::setw(15) << GMaker::ShapeName(shapecode)
                  << " spec " << spec
                  << " boundary " << boundary
                  << " param " << gformat(param)
                  ;

        std::vector<GSolid*> ss = m_maker->make(i, shapecode, param, spec ); 

        for(unsigned int j=0 ; j < ss.size() ; j++)
        {
            GSolid* solid = ss[j];
            solid->setBoundary(boundary);
            GParts* pts = solid->getParts();
            if(pts) pts->setBoundaryAll(boundary);

            solids.push_back(solid);
        } 
    }


    for(unsigned int i=0 ; i < solids.size() ; i++)
    {
        GSolid* solid = solids[i];
        solid->setIndex(i);
        GParts* pts = solid->getParts();
        if(pts)
        { 
            pts->setIndex(0u, solid->getIndex());
            pts->setNodeIndex(0u, solid->getIndex());
            pts->setBndLib(m_bndlib);
        }
    }
    


    GMergedMesh* tri = GMergedMesh::combine( 0, NULL, solids );

    GTransforms* txf = GTransforms::make(n); // identities
    GIds*        aii = GIds::make(n);        // zeros

    tri->setAnalyticInstancedIdentityBuffer(aii->getBuffer());  
    tri->setITransformsBuffer(txf->getBuffer());

    //  OGeo::makeAnalyticGeometry  requires AII and IT buffers to have same item counts
    return tri ; 
} 





