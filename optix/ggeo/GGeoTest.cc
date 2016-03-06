#include "GGeoTest.hh"
#include "GGeoTestConfig.hh"

#include "GVector.hh"
#include "GCache.hh"
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

#include "NLog.hpp"
#include "NSlice.hpp"
#include "GLMFormat.hpp"
#include "stringutil.hpp"

#include <iomanip>

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>

void GGeoTest::init()
{
    m_ggeo = m_cache->getGGeo();
    if(m_ggeo)
    {
        m_geolib = m_ggeo->getGeoLib();
        m_bndlib = m_ggeo->getBndLib();
    }
    else
    {
        LOG(warning) << "GGeoTest::init booting from cache" ; 
        m_geolib = GGeoLib::load(m_cache);
        m_bndlib = GBndLib::load(m_cache, true );
    }
    m_maker = new GMaker(m_cache);
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

    tmm->setGeoCode( analytic ? 'A' : 'T' );  // to OGeo
    if(tmm->getGeoCode() == 'T') 
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


GMergedMesh* GGeoTest::createPmtInBox()
{
    // somewhat dirtily associates analytic geometry with triangulated for the PMT 
    //
    //   * detdesc parsed analytic geometry in GPmt (see pmt-ecd dd.py tree.py etc..)
    //   * instance-1 GMergedMesh 
    //
    //
    // ? assuming single container 
    char shapecode = m_config->getShape(0) ;
    const char* spec = m_config->getBoundary(0);
    glm::vec4 param = m_config->getParameters(0);

    int verbosity = m_config->getVerbosity();


    LOG(info) << "GGeoTest::createPmtInBox " << shapecode << " : " << spec << " " << gformat(param)  ; 

    // this is using prior DYB specific knowledge...
    // mergedMesh-repeat-candidate-1 is the triangulated PMT 5-solids 

    GMergedMesh* mmpmt = m_geolib->getMergedMesh(1);  


    if(verbosity > 1)
    {
        LOG(info) << "GGeoTest::createPmtInBox"
                  << " verbosity " << verbosity 
                  ;

        mmpmt->dumpSolids("GGeoTest::createPmtInBox GMergedMesh::dumpSolids (before:mmpmt) ");
    }



    NSlice* slice = m_config->getSlice();
    GPmt* pmt = GPmt::load( m_cache, m_bndlib, 0, slice );    // pmtIndex:0

    // associating the analytic GPmt with the triangulated GMergedMesh 

    mmpmt->setParts(pmt->getParts());               
    unsigned int index = mmpmt->getNumSolids() ;


    std::vector<GSolid*> solids = m_maker->make( index, shapecode, param, spec) ;

    for(unsigned int j=0 ; j < solids.size() ; j++)
    {
        GSolid* solid = solids[j];
        solid->getMesh()->setIndex(1000);
    }



    GMergedMesh* tri = GMergedMesh::combine( mmpmt->getIndex(), mmpmt, solids );   


    if(verbosity > 1)
        tri->dumpSolids("GGeoTest::createPmtInBox GMergedMesh::dumpSolids (after:tri) ");



    GParts* anl = tri->getParts();
    const char* imat = m_bndlib->getInnerMaterialName(spec);
    anl->setContainingMaterial(imat);   // match outer material of PMT with inner material of the box
    anl->setSensorSurface("lvPmtHemiCathodeSensorSurface") ; // kludge, TODO: investigate where triangulated gets this from
    anl->close();

    tri->setAnalyticInstancedIdentityBuffer(mmpmt->getAnalyticInstancedIdentityBuffer());
    tri->setITransformsBuffer(mmpmt->getITransformsBuffer());

    return tri ; 
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





