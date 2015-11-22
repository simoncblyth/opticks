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
    ;

const char* GGeoTest::MODE_ = "mode"; 
const char* GGeoTest::FRAME_ = "frame"; 
const char* GGeoTest::BOUNDARY_ = "boundary"; 
const char* GGeoTest::PARAMETERS_ = "parameters"; 
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

        set(getArg(it->first.c_str()), it->second.c_str());
    }
}

GGeoTest::Arg_t GGeoTest::getArg(const char* k)
{
    Arg_t arg = UNRECOGNIZED ; 
    if(     strcmp(k,MODE_)==0)       arg = MODE ; 
    else if(strcmp(k,FRAME_)==0)      arg = FRAME ; 
    else if(strcmp(k,BOUNDARY_)==0)   arg = BOUNDARY ; 
    else if(strcmp(k,PARAMETERS_)==0) arg = PARAMETERS ; 
    else if(strcmp(k,SHAPE_)==0)      arg = SHAPE ; 
    else if(strcmp(k,SLICE_)==0)      arg = SLICE ; 
    else if(strcmp(k,ANALYTIC_)==0)   arg = ANALYTIC ; 
    else if(strcmp(k,DEBUG_)==0)      arg = DEBUG ; 
    return arg ;   
}

void GGeoTest::set(Arg_t arg, const char* s)
{
    switch(arg)
    {
        case MODE           : setMode(s)           ;break;
        case FRAME          : setFrame(s)          ;break;
        case BOUNDARY       : addBoundary(s)       ;break;
        case PARAMETERS     : addParameters(s)     ;break;
        case SHAPE          : addShape(s)          ;break;
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

void GGeoTest::setDebug(const char* s)
{
    std::string ss(s);
    m_debug = gvec4(ss);
}
void GGeoTest::addParameters(const char* s)
{
    std::string ss(s);
    m_parameters.push_back(gvec4(ss));
}

void GGeoTest::addBoundary(const char* s)
{
    m_boundaries.push_back(s);
}

void GGeoTest::addShape(const char* s)
{
    m_shapes.push_back(s);
}


void GGeoTest::modifyGeometry()
{
    unsigned int nbnd = m_boundaries.size() ; 
    unsigned int npar = m_parameters.size() ; 
    unsigned int nshp = m_shapes.size() ; 

    assert( nbnd == npar && nbnd == nshp && "need equal number of boundaries, parameters and shapes");
    assert(nbnd > 0);

    GMergedMesh* tmm(NULL);
    // TODO: eliminate the mode unifying createPmtInBox, createBoxInBox -> create
    if(     strcmp(m_mode, "PmtInBox") == 0) tmm = createPmtInBox(); 
    else if(strcmp(m_mode, "BoxInBox") == 0) tmm = createBoxInBox(); 
    else  LOG(warning) << "GGeoTest::modifyGeometry mode not recognized " << m_mode ; 

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


glm::vec4 GGeoTest::getParameters(unsigned int i)
{
    unsigned int npars = m_parameters.size();
    assert( i < npars ) ; 
    glm::vec4 param = m_parameters[i] ;
    return param ;  
}

char GGeoTest::getShape(unsigned int i)
{
    assert( i < m_shapes.size() );
    char shapecode = GMaker::ShapeCode(m_shapes[i].c_str());
    return shapecode ; 
}

const char* GGeoTest::getBoundary(unsigned int i)
{
    assert( i < m_boundaries.size() );
    const char* spec = m_boundaries[i].c_str() ;
    return spec ; 
}



GMergedMesh* GGeoTest::createPmtInBox()
{
    const char* spec = getBoundary(0);
    char shapecode = getShape(0) ;
    glm::vec4 param = getParameters(0);

    LOG(info) << "GGeoTest::createPmtInBox " << shapecode << " : " << spec << " " << gformat(param)  ; 

    GPmt* pmt = GPmt::load( m_cache, m_bndlib, 0, m_slice );    // pmtIndex:0

    GMergedMesh* mmpmt = m_geolib->getMergedMesh(1);  // DYB mesh-1 is the PMT 5-solids 
    mmpmt->setParts(pmt->getParts());  // associate analytic parts with the triangulated PMT

    unsigned int index = mmpmt->getNumSolids() ;

    std::vector<GSolid*> solids = m_maker->make( index, shapecode, param, spec) ;
    for(unsigned int j=0 ; j < solids.size() ; j++)
    {
        GSolid* solid = solids[j];
        solid->getMesh()->setIndex(1000);
    }

    GMergedMesh* tri = GMergedMesh::combine( mmpmt->getIndex(), mmpmt, solids );   

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
        char shapecode = getShape(i) ;
        const char* spec = getBoundary(i);
        glm::vec4 param = getParameters(i);
        unsigned int boundary = m_bndlib->addBoundary(spec);

        LOG(info) << "GGeoTest::createBoxInBox" 
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



