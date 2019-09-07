/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#include <cstddef>
#include <iomanip>
#include <algorithm>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>

#include "BStr.hh"

#include "NGLM.hpp"
#include "GLMFormat.hpp"

#include "NGeoTestConfig.hpp"

#include "PLOG.hh"


//    "pmtpath=$IDPATH_DPIB_PMT/GMergedMesh/0_"


// TODO: OpticksResource::getBasePath("dpib/GMergedMesh/0") 
//       is now used by GPmtLib ... so no need for pmtpath below ?
//

const char* NGeoTestConfig::DEFAULT_CONFIG = 
    "mode=PmtInBox_"
    "pmtpath=$OPTICKSINSTALLPREFIX/opticksdata/export/dpib/GMergedMesh/0_"
    "control=1,0,0,0_"
    "analytic=1_"
    "outerfirst=1_"
    "node=box_"
    "boundary=Rock/NONE/perfectAbsorbSurface/MineralOil_"
    "parameters=0,0,0,300_"
    ;

const char* NGeoTestConfig::MODE_ = "mode"; 
const char* NGeoTestConfig::FRAME_ = "frame"; 
const char* NGeoTestConfig::BOUNDARY_ = "boundary"; 
const char* NGeoTestConfig::PARAMETERS_ = "parameters"; 
const char* NGeoTestConfig::NODE_ = "node"; 
const char* NGeoTestConfig::ANALYTIC_ = "analytic"; 
const char* NGeoTestConfig::DEBUG_ = "debug"; 
const char* NGeoTestConfig::CONTROL_ = "control"; 
const char* NGeoTestConfig::PMTPATH_ = "pmtpath"; 
const char* NGeoTestConfig::TRANSFORM_ = "transform"; 
const char* NGeoTestConfig::CSGPATH_ = "csgpath"; 
const char* NGeoTestConfig::OFFSETS_ = "offsets"; 
const char* NGeoTestConfig::NAME_ = "name"; 
const char* NGeoTestConfig::OUTERFIRST_ = "outerfirst"; 

const char* NGeoTestConfig::AUTOCONTAINER_ = "autocontainer";   
const char* NGeoTestConfig::AUTOOBJECT_ = "autoobject";   
const char* NGeoTestConfig::AUTOEMITCONFIG_ = "autoemitconfig";   
const char* NGeoTestConfig::AUTOSEQMAP_ = "autoseqmap";   


NGeoTestConfig::NGeoTestConfig(const char* config) 
    : 
    m_config(NULL),
    m_mode(NULL),
    m_pmtpath(NULL),
    m_csgpath(NULL),
    m_name(NULL),
    m_autocontainer(NULL),
    m_autoobject(NULL),
    m_autoemitconfig(NULL),
    m_autoseqmap(NULL),
    m_frame(0,0,0,0),
    m_analytic(0,0,0,0),
    m_outerfirst(1,0,0,0),
    m_debug(1.f,0.f,0.f,0.f),
    m_control(0,0,0,0)
{
    init(config);
}

std::vector<std::pair<std::string, std::string> >& NGeoTestConfig::getCfg()
{
    return m_cfg ; 
}


unsigned NGeoTestConfig::getNumBoundaries()
{
    return m_boundaries.size();
}
unsigned NGeoTestConfig::getNumParameters()
{
    return m_parameters.size() ; 
}
unsigned NGeoTestConfig::getNumNodes()
{
    return m_nodes.size() ; 
}
unsigned NGeoTestConfig::getNumTransforms()
{
    return m_transforms.size() ; 
}




bool NGeoTestConfig::getAnalytic()
{
    bool analytic = m_analytic.x > 0 ;
    return analytic ; 
}

bool NGeoTestConfig::getOuterFirst()
{
    return m_outerfirst.x > 0 ;
}



bool NGeoTestConfig::isNCSG()
{
    return m_csgpath != NULL  ; 
}
bool NGeoTestConfig::isPmtInBox()
{
    return strcmp(m_mode, "PmtInBox") == 0 ; 
}
bool NGeoTestConfig::isBoxInBox()
{
    return strcmp(m_mode, "BoxInBox") == 0 ; 
}




const char* NGeoTestConfig::getMode()
{
    return m_mode ; 
}
const char* NGeoTestConfig::getPmtPath()
{
    return m_pmtpath ; 
}
const char* NGeoTestConfig::getCSGPath()
{
    return m_csgpath ; 
}
const char* NGeoTestConfig::getName()
{
    return m_name ; 
}

int NGeoTestConfig::getVerbosity()
{
    return m_control.x  ; 
}







void NGeoTestConfig::init(const char* config)
{
    configure(config);
}

void NGeoTestConfig::configure(const char* config)
{
    LOG(debug) << "NGeoTestConfig::configure" ; 
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

NGeoTestConfig::Arg_t NGeoTestConfig::getArg(const char* k)
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
    else if(strcmp(k,OUTERFIRST_)==0)  arg = OUTERFIRST ; 
    else if(strcmp(k,AUTOCONTAINER_)==0)  arg = AUTOCONTAINER ; 
    else if(strcmp(k,AUTOOBJECT_)==0)  arg = AUTOOBJECT ; 
    else if(strcmp(k,AUTOEMITCONFIG_)==0)  arg = AUTOEMITCONFIG ; 
    else if(strcmp(k,AUTOSEQMAP_)==0)  arg = AUTOSEQMAP ; 

    if(arg == UNRECOGNIZED)
    {
        LOG(warning) << "NGeoTestConfig::getArg UNRECOGNIZED arg " << k ;  
    }

    return arg ;   
}

void NGeoTestConfig::set(Arg_t arg, const char* s)
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
        case OUTERFIRST     : setOuterFirst(s)     ;break;
        case AUTOCONTAINER  : setAutoContainer(s)  ;break;
        case AUTOOBJECT     : setAutoObject(s)     ;break;
        case AUTOEMITCONFIG : setAutoEmitConfig(s)     ;break;
        case AUTOSEQMAP     : setAutoSeqMap(s)     ;break;
        case UNRECOGNIZED   :
             LOG(warning) << "NGeoTestConfig::set WARNING ignoring unrecognized parameter " << s  ;
    }
}



unsigned NGeoTestConfig::getNumElements()
{
    unsigned nbnd = getNumBoundaries();
    unsigned nnod = getNumNodes();
    unsigned npar = getNumParameters();
    unsigned ntra = getNumTransforms();

    bool equal = nbnd == npar && nbnd == nnod && ntra == npar ;

    if(!equal) 
    {
    LOG(fatal) << "NGeoTestConfig::getNumElements"
               << " ELEMENT MISMATCH IN TEST GEOMETRY CONFIGURATION " 
               << " nbnd (boundaries) " << nbnd  
               << " nnod (nodes) " << nnod  
               << " npar (parameters) " << npar  
               << " ntra (transforms) " << ntra
               ; 
    }

    //assert( equal && "need equal number of boundaries, parameters, transforms and nodes");
    //assert(nbnd > 0);
    return equal ? nbnd : 0u ; 
}


void NGeoTestConfig::dump(const char* msg)
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

void NGeoTestConfig::setMode(const char* s)
{
    m_mode = strdup(s);
}
void NGeoTestConfig::setPmtPath(const char* s)
{
    m_pmtpath = strdup(s);
}
void NGeoTestConfig::setCsgPath(const char* s)
{
    m_csgpath = strdup(s);
}
void NGeoTestConfig::setName(const char* s)
{
    m_name = strdup(s);
}



void NGeoTestConfig::setOffsets(const char* s)
{
    BStr::usplit(m_offsets, s, ',' );
}
unsigned NGeoTestConfig::getNumOffsets()
{
    return m_offsets.size();
}
unsigned NGeoTestConfig::getOffset(unsigned idx)
{
    assert(idx < m_offsets.size());
    return m_offsets[idx] ; 
}

bool NGeoTestConfig::isStartOfOptiXPrimitive(unsigned nodeIdx )
{
    return std::find(m_offsets.begin(), m_offsets.end(), nodeIdx) != m_offsets.end() ; 
}




void NGeoTestConfig::setFrame(const char* s)
{
    std::string ss(s);
    m_frame = givec4(ss);
}
void NGeoTestConfig::setAnalytic(const char* s)
{
    std::string ss(s);
    m_analytic = givec4(ss);
}

void NGeoTestConfig::setOuterFirst(const char* s)
{
    std::string ss(s);
    m_outerfirst = givec4(ss);
}



void NGeoTestConfig::setAutoContainer(const char* s)
{
    m_autocontainer = strdup(s) ; 
}
void NGeoTestConfig::setAutoObject(const char* s)
{
    m_autoobject = strdup(s) ; 
}
void NGeoTestConfig::setAutoEmitConfig(const char* s)
{
    m_autoemitconfig = strdup(s) ; 
}
void NGeoTestConfig::setAutoSeqMap(const char* s)
{
    m_autoseqmap = strdup(s) ; 
}


const char* NGeoTestConfig::getAutoContainer() const 
{
    return m_autocontainer ;
}
const char* NGeoTestConfig::getAutoObject() const 
{
    return m_autoobject ;
}
const char* NGeoTestConfig::getAutoEmitConfig() const 
{
    return m_autoemitconfig ;
}
const char* NGeoTestConfig::getAutoSeqMap() const 
{
    return m_autoseqmap ;
}







void NGeoTestConfig::setDebug(const char* s)
{
    std::string ss(s);
    m_debug = gvec4(ss);
}

void NGeoTestConfig::setControl(const char* s)
{
    std::string ss(s);
    m_control = givec4(ss);
}




void NGeoTestConfig::addParameters(const char* s)
{
    std::string ss(s);
    m_parameters.push_back(gvec4(ss));
}

void NGeoTestConfig::addTransform(const char* s)
{
    std::string ss(s == NULL ? "" : s);

    // when adding non-default pop_back first to replace the identity default
    if(!ss.empty() && m_transforms.size() > 0)
    {
        m_transforms.pop_back();
    }

    m_transforms.push_back(gmat4(ss));
}

void NGeoTestConfig::addBoundary(const char* s)
{
    m_boundaries.push_back(s);
}

void NGeoTestConfig::addNode(const char* s)
{
    m_nodes.push_back(s);
}


glm::vec4 NGeoTestConfig::getParameters(unsigned int i)
{
    unsigned int npars = m_parameters.size();
    assert( i < npars ) ; 
    glm::vec4 param = m_parameters[i] ;
    return param ;  
}


glm::mat4 NGeoTestConfig::getTransform(unsigned int i)
{
    unsigned int ntra = m_transforms.size();
    assert( i < ntra ) ; 
    glm::mat4 trans = m_transforms[i] ;
    return trans ;  
}



/*
char NGeoTestConfig::getNode(unsigned int i)
{
    assert( i < m_nodes.size() );
    char nodecode = CSGChar(m_nodes[i].c_str());
    return nodecode ; 
}
*/


OpticksCSG_t NGeoTestConfig::getTypeCode(unsigned int i)
{
    assert( i < m_nodes.size() );
    return CSGTypeCode(m_nodes[i].c_str());
}




std::string NGeoTestConfig::getNodeString(unsigned int i)
{
    assert( i < m_nodes.size() );
    return m_nodes[i] ;
}



const char* NGeoTestConfig::getBoundary(unsigned int i)
{
    assert( i < m_boundaries.size() );
    const char* spec = m_boundaries[i].c_str() ;
    return spec ; 
}


