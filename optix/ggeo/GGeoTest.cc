#include "GGeoTest.hh"

#include "GVector.hh"
#include "GCache.hh"
#include "GGeo.hh"
#include "GGeoLib.hh"
#include "GBndLib.hh"
#include "GMergedMesh.hh"
#include "GPmt.hh"
#include "GSolid.hh"
#include "GTestBox.hh"
#include "GTestSphere.hh"
#include "GItemList.hh"
#include "GParts.hh"
#include "GTransforms.hh"
#include "GIds.hh"

#include "NLog.hpp"
#include "NSlice.hpp"
#include "GLMFormat.hpp"
#include "stringutil.hpp"
#include <boost/lexical_cast.hpp>

const char* GGeoTest::DEFAULT_CONFIG = 
    "mode=PmtInBox_"
    "boundary=Rock///MineralOil_"
    "dimensions=300,0,0,0_"
    ;

const char* GGeoTest::MODE_ = "mode"; 
const char* GGeoTest::FRAME_ = "frame"; 
const char* GGeoTest::DIMENSIONS_ = "dimensions"; 
const char* GGeoTest::BOUNDARY_ = "boundary"; 
const char* GGeoTest::SLICE_ = "slice"; 
const char* GGeoTest::ANALYTIC_ = "analytic"; 

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
}


void GGeoTest::configure(const char* config_)
{
    LOG(debug) << "GGeoTest::configure" ; 
    m_config = config_ ? strdup(config_) : DEFAULT_CONFIG ; 

    m_cfg = ekv_split(m_config,'_',"="); // element-delim, keyval-delim

    for(std::vector<KV>::const_iterator it=m_cfg.begin() ; it!=m_cfg.end() ; it++)
    {
        LOG(debug) 
                  << std::setw(20) << it->first
                  << " : " 
                  << it->second 
                  ;

        set(getParam(it->first.c_str()), it->second.c_str());
    }
}

GGeoTest::Param_t GGeoTest::getParam(const char* k)
{
    Param_t param = UNRECOGNIZED ; 
    if(           strcmp(k,MODE_)==0) param = MODE ; 
    else if(     strcmp(k,FRAME_)==0) param = FRAME ; 
    else if(strcmp(k,DIMENSIONS_)==0) param = DIMENSIONS ; 
    else if(strcmp(k,BOUNDARY_)==0)   param = BOUNDARY ; 
    else if(strcmp(k,SLICE_)==0)      param = SLICE ; 
    else if(strcmp(k,ANALYTIC_)==0)   param = ANALYTIC ; 
    return param ;   
}

void GGeoTest::set(Param_t p, const char* s)
{
    switch(p)
    {
        case MODE           : setMode(s)           ;break;
        case FRAME          : setFrame(s)          ;break;
        case DIMENSIONS     : setDimensions(s)     ;break;
        case BOUNDARY       : addBoundary(s)       ;break;
        case SLICE          : setSlice(s)          ;break;
        case ANALYTIC       : setAnalytic(s)       ;break;
        case UNRECOGNIZED   :
                    LOG(warning) << "GGeoTest::set WARNING ignoring unrecognized parameter " ;
    }
}

void GGeoTest::dump(const char* msg)
{
    LOG(info) << msg  
              << " config " << m_config 
              << " mode " << m_mode 
              ; 
}

void GGeoTest::setMode(const char* s)
{
    m_mode = strdup(s);
}
void GGeoTest::setSlice(const char* s)
{
    m_slice = s ? new NSlice(s) : NULL ;
}
void GGeoTest::setFrame(const char* s)
{
    std::string ss(s);
    m_frame = givec4(ss);
}
void GGeoTest::setAnalytic(const char* s)
{
    std::string ss(s);
    m_analytic = givec4(ss);
}
void GGeoTest::setDimensions(const char* s)
{
    std::string ss(s);
    m_dimensions = gvec4(ss);
}
void GGeoTest::addBoundary(const char* s)
{
    m_boundaries.push_back(s);
}



void GGeoTest::modifyGeometry()
{
    GMergedMesh* tmm(NULL);

    if(strcmp(m_mode, "PmtInBox") == 0)
    {
         assert(m_dimensions.x  > 0);
         assert(m_boundaries.size() > 0);
         tmm = createPmtInBox(); 
    }
    else if(strcmp(m_mode, "BoxInBox") == 0)
    {
         unsigned int n = m_boundaries.size() ;
         assert(n > 0 && n < 4);
         tmm = createBoxInBox(); 
    }
    else
    {
         LOG(warning) << "GGeoTest::modifyGeometry mode not recognized " << m_mode ; 
    }

    if(!tmm) return ; 

    //tmm->dump("GGeoTest::modifyGeometry tmm ");
    m_geolib->clear();
    m_geolib->setMergedMesh( 0, tmm );
}



GMergedMesh* GGeoTest::createPmtInBox()
{
    float size = m_dimensions.x ;
    const char* spec = m_boundaries[0].c_str() ;
    unsigned int boundary = m_bndlib->addBoundary(spec);
    const char* imat = m_bndlib->getInnerMaterialName(boundary);

    LOG(info) << "GGeoTest::createPmtInBox" 
              << " container size  " << size
              << " spec " << spec
              << " boundary " << boundary
              << " imat " << imat
              ; 

    // still using mesh to set container box size basis for the analytic...
    GMergedMesh* mm = m_geolib->getMergedMesh(1);
    //gbbox bb = mm->getBBox(0);      // solid-0 contains them all
    //bb->enlarge(size);   // THIS OLD APPROACH SHOWS BLACK EDGED BOX OPTIX RENDER, NOT SEED WITH ABSOLUTE DIMENSIONS

    gbbox bb(gfloat3(-size), gfloat3(size));  


    GPmt* pmt = GPmt::load( m_cache, m_bndlib, 0, m_slice );    // pmtIndex:0

    GParts* ppmt = pmt->getParts();
    ppmt->setContainingMaterial(imat);   // match outer material of PMT with inner material of the box
    ppmt->add(GParts::makeBox(bb, spec));
    ppmt->setSensorSurface("lvPmtHemiCathodeSensorSurface") ; // kludge, TODO: investigate where triangulated gets this from
    ppmt->close();


    // triangulated setup 

    GSolid* solid = GTestBox::makeSolid(bb, 1000, mm->getNumSolids()) ; // meshIndex, nodeIndex
    solid->setBoundary(boundary); 

    GMergedMesh* pib = GMergedMesh::combine( mm->getIndex(), mm, solid );   
    //pib->dump("GGeoTest::createPmtInBox");

    pib->setGeoCode('S');  // signal OGeo to use Analytic geometry
    pib->setParts(ppmt);
    pib->setAnalyticInstancedIdentityBuffer(mm->getAnalyticInstancedIdentityBuffer());
    pib->setITransformsBuffer(mm->getITransformsBuffer());


    if( pib->getNumSolids() != ppmt->getNumSolids() )
        LOG(warning) << "GGeoTest::createPmtInBox"
                     << " analytic/triangulated solid count mismatch "
                     << " triangulated " << pib->getNumSolids()
                     << " analytic " << ppmt->getNumSolids()
                     ;


    return pib ; 
}



GMergedMesh* GGeoTest::createBoxInBox()
{
    std::vector<GSolid*> solids ; 
    GParts* analytic = new GParts(m_bndlib);

    GTransforms* transforms = new GTransforms();
    GIds* aii = new GIds();


    unsigned int n = m_boundaries.size();
    for(unsigned int i=0 ; i < n ; i++)
    {
        float s = m_dimensions[i] ;
        gbbox bb(gfloat3(-s), gfloat3(s));  
        glm::vec4 sphere(0.f,0.f,0.f,s) ;

        //GSolid* solid = GTestBox::makeSolid(bb, 1000, i ) ; // meshIndex, nodeIndex
        GSolid* solid = GTestSphere::makeSolid(sphere, 1000, i ) ; // meshIndex, nodeIndex

        const char* spec = m_boundaries[i].c_str() ;
        unsigned int boundary = m_bndlib->addBoundary(spec);
        solid->setBoundary(boundary);
        solid->setSensor(NULL);
        solids.push_back(solid);

        transforms->add();
        aii->add(0,0,0,0);  // placeholder

        analytic->add(GParts::makeBox(bb, spec));

        LOG(info) << "GGeoTest::createBoxInBox"
                  << " solid " << std::setw(2) << i 
                  << " boundary " << std::setw(3) << boundary
                  << " spec " << spec
                  << bb.description() 
                  ;
    }

    GMergedMesh* bib = GMergedMesh::combine( 0, NULL, solids );
    bib->setParts(analytic);


    bib->setAnalyticInstancedIdentityBuffer(aii->getBuffer());  
    bib->setITransformsBuffer(transforms->getBuffer());
    //  OGeo::makeAnalyticGeometry  requires AII and IT buffers to have same item counts


    if(m_analytic.x > 0)
    {
        LOG(info) << "GGeoTest::createBoxInBox using analytic " ; 
        bib->setGeoCode('S');  // signal OGeo to use Analytic geometry
    }
    else
    {
        bib->setITransformsBuffer(NULL); // run into FaceRepeated complications when non-NULL
        bib->setGeoCode('T');  
        LOG(info) << "GGeoTest::createBoxInBox using triangulated " ; 
    }

    return bib ; 
} 


