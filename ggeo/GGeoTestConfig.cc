#include <cstddef>
#include <iomanip>
#include <algorithm>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>

#include "BStr.hh"

#include "NGLM.hpp"
#include "GLMFormat.hpp"

#include "GMaker.hh"
#include "GGeoTestConfig.hh"

#include "PLOG.hh"


//    "pmtpath=$IDPATH_DPIB_PMT/GMergedMesh/0_"

const char* GGeoTestConfig::DEFAULT_CONFIG = 
    "mode=PmtInBox_"
    "pmtpath=$OPTICKSINSTALLPREFIX/opticksdata/export/dpib/GMergedMesh/0_"
    "control=1,0,0,0_"
    "analytic=1_"
    "node=box_"
    "boundary=Rock/NONE/perfectAbsorbSurface/MineralOil_"
    "parameters=0,0,0,300_"
    ;

const char* GGeoTestConfig::MODE_ = "mode"; 
const char* GGeoTestConfig::FRAME_ = "frame"; 
const char* GGeoTestConfig::BOUNDARY_ = "boundary"; 
const char* GGeoTestConfig::PARAMETERS_ = "parameters"; 
const char* GGeoTestConfig::NODE_ = "node"; 
const char* GGeoTestConfig::ANALYTIC_ = "analytic"; 
const char* GGeoTestConfig::DEBUG_ = "debug"; 
const char* GGeoTestConfig::CONTROL_ = "control"; 
const char* GGeoTestConfig::PMTPATH_ = "pmtpath"; 
const char* GGeoTestConfig::TRANSFORM_ = "transform"; 
const char* GGeoTestConfig::CSGPATH_ = "csgpath"; 
const char* GGeoTestConfig::OFFSETS_ = "offsets"; 
const char* GGeoTestConfig::NAME_ = "name"; 




GGeoTestConfig::GGeoTestConfig(const char* config) 
    : 
    m_config(NULL),
    m_mode(NULL),
    m_pmtpath(NULL),
    m_csgpath(NULL),
    m_name(NULL),
    m_frame(0,0,0,0),
    m_analytic(0,0,0,0),
    m_debug(1.f,0.f,0.f,0.f),
    m_control(0,0,0,0)
{
    init(config);
}

std::vector<std::pair<std::string, std::string> >& GGeoTestConfig::getCfg()
{
    return m_cfg ; 
}


unsigned GGeoTestConfig::getNumBoundaries()
{
    return m_boundaries.size();
}
unsigned GGeoTestConfig::getNumParameters()
{
    return m_parameters.size() ; 
}
unsigned GGeoTestConfig::getNumNodes()
{
    return m_nodes.size() ; 
}
unsigned GGeoTestConfig::getNumTransforms()
{
    return m_transforms.size() ; 
}




bool GGeoTestConfig::getAnalytic()
{
    bool analytic = m_analytic.x > 0 ;
    return analytic ; 
}


const char* GGeoTestConfig::getMode()
{
    return m_mode ; 
}
const char* GGeoTestConfig::getPmtPath()
{
    return m_pmtpath ; 
}
const char* GGeoTestConfig::getCsgPath()
{
    return m_csgpath ; 
}
const char* GGeoTestConfig::getName()
{
    return m_name ; 
}

int GGeoTestConfig::getVerbosity()
{
    return m_control.x  ; 
}







void GGeoTestConfig::init(const char* config)
{
    configure(config);
}

void GGeoTestConfig::configure(const char* config)
{
    LOG(debug) << "GGeoTestConfig::configure" ; 
    m_config = config ? strdup(config) : DEFAULT_CONFIG ; 

    m_cfg = BStr::ekv_split(m_config,'_',"="); // element-delim, keyval-delim

    for(std::vector<KV>::const_iterator it=m_cfg.begin() ; it!=m_cfg.end() ; it++)
    {
        LOG(debug) 
                  << std::setw(20) << it->first
                  << " : " 
                  << it->second 
                  ;

        Arg_t arg = getArg(it->first.c_str()) ;

        set(arg, it->second.c_str());
        if(arg == NODE)
        {
            set(TRANSFORM, NULL); // default transform for each "shape" is identity  ("shape" in becoming "node") 
        }
    }
}

GGeoTestConfig::Arg_t GGeoTestConfig::getArg(const char* k)
{
    Arg_t arg = UNRECOGNIZED ; 
    if(     strcmp(k,MODE_)==0)       arg = MODE ; 
    else if(strcmp(k,FRAME_)==0)      arg = FRAME ; 
    else if(strcmp(k,BOUNDARY_)==0)   arg = BOUNDARY ; 
    else if(strcmp(k,PARAMETERS_)==0) arg = PARAMETERS ; 
    else if(strcmp(k,NODE_)==0)       arg = NODE ; 
    else if(strcmp(k,ANALYTIC_)==0)   arg = ANALYTIC ; 
    else if(strcmp(k,DEBUG_)==0)      arg = DEBUG ; 
    else if(strcmp(k,CONTROL_)==0)    arg = CONTROL ; 
    else if(strcmp(k,PMTPATH_)==0)    arg = PMTPATH ; 
    else if(strcmp(k,TRANSFORM_)==0)  arg = TRANSFORM ; 
    else if(strcmp(k,CSGPATH_)==0)    arg = CSGPATH ; 
    else if(strcmp(k,OFFSETS_)==0)    arg = OFFSETS ; 
    else if(strcmp(k,NAME_)==0)       arg = NAME ; 

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
        case NODE           : addNode(s)           ;break;
        case ANALYTIC       : setAnalytic(s)       ;break;
        case DEBUG          : setDebug(s)          ;break;
        case CONTROL        : setControl(s)        ;break;
        case PMTPATH        : setPmtPath(s)        ;break;
        case TRANSFORM      : addTransform(s)      ;break;
        case CSGPATH        : setCsgPath(s)        ;break;
        case OFFSETS        : setOffsets(s)        ;break;
        case NAME           : setName(s)           ;break;
        case UNRECOGNIZED   :
             LOG(warning) << "GGeoTestConfig::set WARNING ignoring unrecognized parameter " << s  ;
    }
}



unsigned GGeoTestConfig::getNumElements()
{
    unsigned nbnd = getNumBoundaries();
    unsigned nnod = getNumNodes();
    unsigned npar = getNumParameters();
    unsigned ntra = getNumTransforms();

    bool equal = nbnd == npar && nbnd == nnod && ntra == npar ;

    if(!equal) 
    LOG(fatal) << "GGeoTestConfig::getNumElements"
               << " ELEMENT MISMATCH IN TEST GEOMETRY CONFIGURATION " 
               << " nbnd (boundaries) " << nbnd  
               << " nnod (nodes) " << nnod  
               << " npar (parameters) " << npar  
               << " ntra (transforms) " << ntra
               ; 

    assert( equal && "need equal number of boundaries, parameters, transforms and nodes");
    //assert(nbnd > 0);
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
        //char csgChar = getNode(i) ;
        OpticksCSG_t type = getTypeCode(i) ;
        const char* spec = getBoundary(i);
        glm::vec4 param = getParameters(i);

        std::cout
                  << " i " << std::setw(2) << i 
                  << " type " << std::setw(2) << type
                  << " csgName " << std::setw(15) << CSGName(type)
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
void GGeoTestConfig::setCsgPath(const char* s)
{
    m_csgpath = strdup(s);
}
void GGeoTestConfig::setName(const char* s)
{
    m_name = strdup(s);
}



void GGeoTestConfig::setOffsets(const char* s)
{
    BStr::usplit(m_offsets, s, ',' );
}
unsigned GGeoTestConfig::getNumOffsets()
{
    return m_offsets.size();
}
unsigned GGeoTestConfig::getOffset(unsigned idx)
{
    assert(idx < m_offsets.size());
    return m_offsets[idx] ; 
}

bool GGeoTestConfig::isStartOfOptiXPrimitive(unsigned nodeIdx )
{
    return std::find(m_offsets.begin(), m_offsets.end(), nodeIdx) != m_offsets.end() ; 
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

void GGeoTestConfig::addTransform(const char* s)
{
    std::string ss(s == NULL ? "" : s);

    // when adding non-default pop_back first to replace the identity default
    if(!ss.empty() && m_transforms.size() > 0)
    {
        m_transforms.pop_back();
    }

    m_transforms.push_back(gmat4(ss));
}

void GGeoTestConfig::addBoundary(const char* s)
{
    m_boundaries.push_back(s);
}

void GGeoTestConfig::addNode(const char* s)
{
    m_nodes.push_back(s);
}


glm::vec4 GGeoTestConfig::getParameters(unsigned int i)
{
    unsigned int npars = m_parameters.size();
    assert( i < npars ) ; 
    glm::vec4 param = m_parameters[i] ;
    return param ;  
}


glm::mat4 GGeoTestConfig::getTransform(unsigned int i)
{
    unsigned int ntra = m_transforms.size();
    assert( i < ntra ) ; 
    glm::mat4 trans = m_transforms[i] ;
    return trans ;  
}



/*
char GGeoTestConfig::getNode(unsigned int i)
{
    assert( i < m_nodes.size() );
    char nodecode = CSGChar(m_nodes[i].c_str());
    return nodecode ; 
}
*/


OpticksCSG_t GGeoTestConfig::getTypeCode(unsigned int i)
{
    assert( i < m_nodes.size() );
    return CSGTypeCode(m_nodes[i].c_str());
}




std::string GGeoTestConfig::getNodeString(unsigned int i)
{
    assert( i < m_nodes.size() );
    return m_nodes[i] ;
}



const char* GGeoTestConfig::getBoundary(unsigned int i)
{
    assert( i < m_boundaries.size() );
    const char* spec = m_boundaries[i].c_str() ;
    return spec ; 
}


