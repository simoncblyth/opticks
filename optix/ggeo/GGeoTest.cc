#include "GGeoTest.hh"

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

const char* GGeoTest::DEFAULT_CONFIG = 
    "mode=PmtInBox_"
    "boundary=Rock///MineralOil_"
    "dimensions=300,0,0,0_"
    ;

const char* GGeoTest::MODE_ = "mode"; 
const char* GGeoTest::FRAME_ = "frame"; 
const char* GGeoTest::DIMENSIONS_ = "dimensions"; 
const char* GGeoTest::BOUNDARY_ = "boundary"; 
const char* GGeoTest::SHAPE_ = "shape"; 
const char* GGeoTest::SLICE_ = "slice"; 
const char* GGeoTest::ANALYTIC_ = "analytic"; 
const char* GGeoTest::DEBUG_ = "debug"; 


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
    if(     strcmp(k,MODE_)==0)       param = MODE ; 
    else if(strcmp(k,FRAME_)==0)      param = FRAME ; 
    else if(strcmp(k,DIMENSIONS_)==0) param = DIMENSIONS ; 
    else if(strcmp(k,BOUNDARY_)==0)   param = BOUNDARY ; 
    else if(strcmp(k,SHAPE_)==0)      param = SHAPE ; 
    else if(strcmp(k,SLICE_)==0)      param = SLICE ; 
    else if(strcmp(k,ANALYTIC_)==0)   param = ANALYTIC ; 
    else if(strcmp(k,DEBUG_)==0)      param = DEBUG ; 
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
        case SHAPE          : setShape(s)          ;break;
        case SLICE          : setSlice(s)          ;break;
        case ANALYTIC       : setAnalytic(s)       ;break;
        case DEBUG          : setDebug(s)          ;break;
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
void GGeoTest::setDebug(const char* s)
{
    std::string ss(s);
    m_debug = gvec4(ss);
}
void GGeoTest::addBoundary(const char* s)
{
    m_boundaries.push_back(s);
}
void GGeoTest::setShape(const char* s)
{
    std::vector<std::string> elem ; 
    boost::split(elem, s, boost::is_any_of(","));    
    unsigned int n = elem.size();
    for(unsigned int k=0 ; k < 4 ; k++)
    {
       if( n > k) m_shape[k] = boost::lexical_cast<char>(elem[k]);     
    }

    LOG(info) << "GGeoTest::setShape " 
              << " x " << (char)m_shape.x
              << " y " << (char)m_shape.y
              << " z " << (char)m_shape.z
              << " w " << (char)m_shape.w
              ;
}




void GGeoTest::modifyGeometry()
{
    GMergedMesh* tmm(NULL);

    // TODO: eliminate the mode unifying createPmtInBox, createBoxInBox -> create

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


    bool analytic = m_analytic.x > 0 ;

    tmm->setGeoCode( analytic ? 'A' : 'T' );  // to OGeo
    if(tmm->getGeoCode() == 'T') 
    { 
        tmm->setITransformsBuffer(NULL); // avoiding FaceRepeated complications 
    } 

    //tmm->dump("GGeoTest::modifyGeometry tmm ");
    m_geolib->clear();
    m_geolib->setMergedMesh( 0, tmm );
}



GMergedMesh* GGeoTest::createPmtInBox()
{
    const char* spec = m_boundaries[0].c_str() ;
    char shapecode = (char)m_shape[0] ;

    LOG(info) << "GGeoTest::createPmtInBox" << " spec " << spec ; 

    GPmt* pmt = GPmt::load( m_cache, m_bndlib, 0, m_slice );    // pmtIndex:0
    GMergedMesh* mmpmt = m_geolib->getMergedMesh(1);  // DYB mesh-1 is the PMT 5-solids 
    //TODO: some kinda matching sanity check between analytic and triangulated ? 
    mmpmt->setParts(pmt->getParts());  // associate analytic parts with the triangulated PMT


    unsigned int index = mmpmt->getNumSolids() ;

    glm::vec4 param(0.f,0.f,0.f,m_dimensions[0]) ;
    GSolid* solid = m_maker->make( index, shapecode, param, spec) ;
    solid->getMesh()->setIndex(1000);

    GMergedMesh* tri = GMergedMesh::combine( mmpmt->getIndex(), mmpmt, solid );   

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
    unsigned int n = m_boundaries.size();

    for(unsigned int i=0 ; i < n ; i++)
    {
        char   shapecode = (char)m_shape[i] ;
        const char* spec = m_boundaries[i].c_str() ;
        bool zsp = shapecode == 'Z' ;

        LOG(info) << "GGeoTest::createBoxInBox" 
                  << " i " << std::setw(2) << i 
                  << " shapecode " << std::setw(2) << shapecode 
                  << " shapename " << std::setw(15) << GMaker::ShapeName(shapecode)
                  << " spec " << spec
                  ;
        
        glm::vec4 param(zsp ? 0.5 : 0.f,zsp ? 1.0 : 0.f,0.f,m_dimensions[i]) ;
        GSolid* solid = m_maker->make(i, shapecode, param, spec ); 
        solid->setSensor(NULL);

        solids.push_back(solid);
    }

    GMergedMesh* tri = GMergedMesh::combine( 0, NULL, solids );

    GTransforms* txf = GTransforms::make(n); // identities
    GIds*        aii = GIds::make(n);        // zeros

    tri->setAnalyticInstancedIdentityBuffer(aii->getBuffer());  
    tri->setITransformsBuffer(txf->getBuffer());

    //  OGeo::makeAnalyticGeometry  requires AII and IT buffers to have same item counts
    return tri ; 
} 



