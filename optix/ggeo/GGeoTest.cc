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
#include "GItemList.hh"

#include "NLog.hpp"
#include "GLMFormat.hpp"
#include "stringutil.hpp"
#include <boost/lexical_cast.hpp>

const char* GGeoTest::DEFAULT_CONFIG = 
    "mode=PmtInBox_"
    "boundary=Rock///MineralOil_"
    "dimensions=3,0,0,0_"
    ;

const char* GGeoTest::MODE_ = "mode"; 
const char* GGeoTest::FRAME_ = "frame"; 
const char* GGeoTest::DIMENSIONS_ = "dimensions"; 
const char* GGeoTest::BOUNDARY_ = "boundary"; 

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

    std::string config(m_config);
    typedef std::pair<std::string,std::string> KV ; 
    std::vector<KV> ekv = ekv_split(config.c_str(),'_',"="); // element-delim, keyval-delim


    for(std::vector<KV>::const_iterator it=ekv.begin() ; it!=ekv.end() ; it++)
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

    m_bndlib->dumpBoundaries(m_boundaries, "GGeoTest::dump");
}

void GGeoTest::setMode(const char* s)
{
    m_mode = strdup(s);
}
void GGeoTest::setFrame(const char* s)
{
    std::string ss(s);
    m_frame = givec4(ss);
}
void GGeoTest::setDimensions(const char* s)
{
    std::string ss(s);
    m_dimensions = gvec4(ss);
}
void GGeoTest::addBoundary(const char* s)
{
    unsigned int boundary = m_bndlib->addBoundary(s); 
    LOG(info) << "GGeoTest::addBoundary " << s << " : " << boundary ; 
    m_boundaries.push_back(boundary);
}



void GGeoTest::modifyGeometry()
{
    GMergedMesh* tmm(NULL);

    if(strcmp(m_mode, "PmtInBox") == 0)
    {
         assert(m_dimensions.x  > 0);
         assert(m_boundaries.size() > 0);
         tmm = createPmtInBox(m_dimensions.x, m_boundaries[0]); 
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



GMergedMesh* GGeoTest::createPmtInBox(float size, unsigned int boundary)
{
    LOG(info) << "GGeoTest::createPmtInBox" ; 

    GMergedMesh* mm = m_geolib->getMergedMesh(1);

    GPmt* pmt = GPmt::load( m_cache, 0, NULL );  // part slicing disfavored, as only works at one level 
    pmt->dump();

    GItemList* bndspec = pmt->getBndSpec();
    unsigned int nbnd = bndspec->getNumKeys();

    for(unsigned int i=0 ; i < nbnd ; i++)
    {
        const char* bnd = bndspec->getKey(i);
        LOG(info) << "GGeoTest::createPmtInBox"
                  << std::setw(3) << i 
                  << " : " << bnd 
                  ; 

        if(strncmp(bnd, GPmt::OUTERMATERIAL, strlen(GPmt::OUTERMATERIAL)) == 0)
        {
            LOG(info) << "starts with marker" ; 
        }


    }


    // TODO: break this requirement, GPmt needs to manage its own identities...
    assert( pmt->getNumSolids() == mm->getNumSolids() );

    gbbox bb = mm->getBBox(0);     // solid-0 contains them all
    bb.enlarge(size);               // the **ONE** place for sizing containment box

    GSolid* solid = GTestBox::makeSolid(bb, 1000, mm->getNumSolids()) ; // meshIndex, nodeIndex
    solid->setBoundary(boundary); 

    GMergedMesh* pib = GMergedMesh::combine( mm->getIndex(), mm, solid );   
    pib->dump();
    pib->setGeoCode('S');  // signal OGeo to use Analytic geometry, TODO: set it not signal it  
    pib->setPmt(pmt);

    unsigned int nodeindex = pmt->getNumSolids();
    pmt->addContainer(bb, nodeindex );
    pmt->dump("after addContainer");

    assert( pib->getNumSolids() == pmt->getNumSolids() );

    return pib ; 
}



GMergedMesh* GGeoTest::createBoxInBox()
{
    std::vector<GSolid*> solids ; 

    unsigned int n = m_boundaries.size();
    for(unsigned int i=0 ; i < n ; i++)
    {
        float s = m_dimensions[i] ;
        gbbox bb(gfloat3(-s), gfloat3(s));  

        GSolid* solid = GTestBox::makeSolid(bb, 1000, i ) ; // meshIndex, nodeIndex
        unsigned int boundary = m_boundaries[i];
        solid->setBoundary(boundary);
        solid->setSensor(NULL);
        solids.push_back(solid);

        LOG(info) << "GGeoTest::createBoxInBox"
                  << " solid " << std::setw(2) << i 
                  << " boundary " << std::setw(3) << boundary
                  << bb.description() 
                  ;
    }

    GMergedMesh* bib = GMergedMesh::combine( 0, NULL, solids );
    return bib ; 
} 


