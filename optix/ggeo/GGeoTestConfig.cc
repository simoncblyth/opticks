#include "GGeoTestConfig.hh"

#include "GMaker.hh"

#include "BLog.hh"
#include "NSlice.hpp"
#include "GLMFormat.hpp"
#include "stringutil.hpp"

#include <iomanip>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>

const char* GGeoTestConfig::DEFAULT_CONFIG = 
    "mode=PmtInBox_"
    "pmtpath=$IDPATH_DPIB_PMT/GMergedMesh/0_"
    "control=1,0,0,0_"
    "analytic=1_"
    "groupvel=0_"
    "shape=box_"
    "boundary=Rock//perfectAbsorbSurface/MineralOil_"
    "parameters=0,0,0,300_"
    ;

const char* GGeoTestConfig::MODE_ = "mode"; 
const char* GGeoTestConfig::FRAME_ = "frame"; 
const char* GGeoTestConfig::BOUNDARY_ = "boundary"; 
const char* GGeoTestConfig::PARAMETERS_ = "parameters"; 
const char* GGeoTestConfig::SHAPE_ = "shape"; 
const char* GGeoTestConfig::SLICE_ = "slice"; 
const char* GGeoTestConfig::ANALYTIC_ = "analytic"; 
const char* GGeoTestConfig::DEBUG_ = "debug"; 
const char* GGeoTestConfig::CONTROL_ = "control"; 
const char* GGeoTestConfig::PMTPATH_ = "pmtpath"; 
const char* GGeoTestConfig::GROUPVEL_ = "groupvel"; 

void GGeoTestConfig::init(const char* config)
{
    configure(config);
}

void GGeoTestConfig::configure(const char* config)
{
    LOG(debug) << "GGeoTestConfig::configure" ; 
    m_config = config ? strdup(config) : DEFAULT_CONFIG ; 

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

GGeoTestConfig::Arg_t GGeoTestConfig::getArg(const char* k)
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
    else if(strcmp(k,CONTROL_)==0)    arg = CONTROL ; 
    else if(strcmp(k,PMTPATH_)==0)    arg = PMTPATH ; 
    else if(strcmp(k,GROUPVEL_)==0)   arg = GROUPVEL ; 

    if(arg == UNRECOGNIZED)
        LOG(warning) << "GGeoTestConfig::getArg UNRECOGNIZED arg " << k ; 

    return arg ;   
}

void GGeoTestConfig::set(Arg_t arg, const char* s)
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
        case CONTROL        : setControl(s)        ;break;
        case PMTPATH        : setPmtPath(s)        ;break;
        case GROUPVEL       : setGroupvel(s)       ;break;
        case UNRECOGNIZED   :
             LOG(warning) << "GGeoTestConfig::set WARNING ignoring unrecognized parameter " << s  ;
    }
}



unsigned int GGeoTestConfig::getNumElements()
{
    unsigned int nbnd = getNumBoundaries();
    unsigned int nshp = getNumShapes();
    unsigned int npar = getNumParameters();

    assert( nbnd == npar && nbnd == nshp && "need equal number of boundaries, parameters and shapes");
    assert(nbnd > 0);
    return nbnd ; 
}


void GGeoTestConfig::dump(const char* msg)
{
    unsigned int n = getNumElements();
    LOG(info) << msg  
              << " config " << m_config 
              << " mode " << m_mode 
              << " nelem " << n 
              ; 

    for(unsigned int i=0 ; i < n ; i++)
    {
        char shapecode = getShape(i) ;
        const char* spec = getBoundary(i);
        glm::vec4 param = getParameters(i);

        std::cout
                  << " i " << std::setw(2) << i 
                  << " shapecode " << std::setw(2) << shapecode 
                  << " shapename " << std::setw(15) << GMaker::ShapeName(shapecode)
                  << " param " << std::setw(50) << gformat(param)
                  << " spec " << std::setw(30) << spec
                  << std::endl 
                  ;
    }
}

void GGeoTestConfig::setMode(const char* s)
{
    m_mode = strdup(s);
}
void GGeoTestConfig::setPmtPath(const char* s)
{
    m_pmtpath = strdup(s);
}
void GGeoTestConfig::setSlice(const char* s)
{
    m_slice = s ? new NSlice(s) : NULL ;
}
void GGeoTestConfig::setFrame(const char* s)
{
    std::string ss(s);
    m_frame = givec4(ss);
}
void GGeoTestConfig::setAnalytic(const char* s)
{
    std::string ss(s);
    m_analytic = givec4(ss);
}
void GGeoTestConfig::setGroupvel(const char* s)
{
    std::string ss(s);
    m_groupvel = givec4(ss);
}


void GGeoTestConfig::setDebug(const char* s)
{
    std::string ss(s);
    m_debug = gvec4(ss);
}

void GGeoTestConfig::setControl(const char* s)
{
    std::string ss(s);
    m_control = givec4(ss);
}



void GGeoTestConfig::addParameters(const char* s)
{
    std::string ss(s);
    m_parameters.push_back(gvec4(ss));
}

void GGeoTestConfig::addBoundary(const char* s)
{
    m_boundaries.push_back(s);
}

void GGeoTestConfig::addShape(const char* s)
{
    m_shapes.push_back(s);
}


glm::vec4 GGeoTestConfig::getParameters(unsigned int i)
{
    unsigned int npars = m_parameters.size();
    assert( i < npars ) ; 
    glm::vec4 param = m_parameters[i] ;
    return param ;  
}

char GGeoTestConfig::getShape(unsigned int i)
{
    assert( i < m_shapes.size() );
    char shapecode = GMaker::ShapeCode(m_shapes[i].c_str());
    return shapecode ; 
}

const char* GGeoTestConfig::getBoundary(unsigned int i)
{
    assert( i < m_boundaries.size() );
    const char* spec = m_boundaries[i].c_str() ;
    return spec ; 
}


