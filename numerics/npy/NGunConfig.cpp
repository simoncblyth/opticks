#include <vector>
#include <iomanip>
#include <sstream>
#include <cstring>
#include <cassert>


#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>

// brap-
#include "BStr.hh"

// npy-
#include "NGunConfig.hpp"
#include "GLMPrint.hpp"
#include "GLMFormat.hpp"
#include "uif.h"


#include "PLOG.hh"

NGunConfig::NGunConfig()
    :
     m_config(NULL),
     m_comment(NULL),
     m_particle(NULL),
     m_number(1),
     m_time(0.),
     m_energy(0.)
{
    init();
}


void NGunConfig::setTime(float time)
{
    m_time = time ; 
}
void NGunConfig::setEnergy(float energy)
{
    m_energy = energy ; 
}
void NGunConfig::setNumber(unsigned int number)
{
    m_number = number ; 
}


void NGunConfig::setFrameTransform(glm::mat4& frame_transform)
{
    m_frame_transform = frame_transform ;
    update();
}
const glm::mat4& NGunConfig::getFrameTransform()
{
    return m_frame_transform ;
}

void NGunConfig::setPosition(const glm::vec3& pos )
{
    m_position = pos ; 
}
void NGunConfig::setDirection(const glm::vec3& dir )
{
    m_direction = dir ; 
}
void NGunConfig::setPolarization(const glm::vec3& pol )
{
    m_polarization = pol ; 
}


const char* NGunConfig::getComment()
{
    return m_comment ; 
}
const char* NGunConfig::getParticle()
{
    return m_particle ; 
}

float NGunConfig::getTime()
{
    return m_time ; 
}
float NGunConfig::getEnergy()
{
    return m_energy ; 
}

int NGunConfig::getFrame()
{
    return m_frame.x ; 
}
unsigned int NGunConfig::getNumber()
{
    return m_number ; 
}



glm::vec3 NGunConfig::getPosition()
{
    return m_position ; 
}
glm::vec3 NGunConfig::getDirection()
{
    return m_direction ; 
}
glm::vec3 NGunConfig::getPolarization()
{
    return m_polarization ; 
}




const char* NGunConfig::DEFAULT_CONFIG = 
    "comment=default-config-comment-without-spaces-_"
    "particle=e+_"
    "frame=3153_"
    "position=0,0,0_"
    "direction=0,0,1_"
    "polarization=1,0,0_"
    "time=0.1_"
    "energy=1.0_"
    "number=1_"
    ;  // mm,ns,MeV 

const char* NGunConfig::COMMENT_ = "comment"; 
const char* NGunConfig::PARTICLE_ = "particle"; 
const char* NGunConfig::FRAME_ = "frame"; 
const char* NGunConfig::POSITION_ = "position"; 
const char* NGunConfig::DIRECTION_ = "direction" ; 
const char* NGunConfig::POLARIZATION_ = "polarization" ; 
const char* NGunConfig::TIME_     = "time" ; 
const char* NGunConfig::ENERGY_   = "energy" ; 
const char* NGunConfig::NUMBER_   = "number" ; 
const char* NGunConfig::UNRECOGNIZED_ = "unrecognized" ; 

NGunConfig::Param_t NGunConfig::parseParam(const char* k)
{
    Param_t param = UNRECOGNIZED ; 
    if(     strcmp(k,COMMENT_)==0)        param = COMMENT ; 
    else if(strcmp(k,PARTICLE_)==0)       param = PARTICLE ; 
    else if(strcmp(k,FRAME_)==0)          param = FRAME ; 
    else if(strcmp(k,POSITION_)==0)       param = POSITION ; 
    else if(strcmp(k,DIRECTION_)==0)      param = DIRECTION ; 
    else if(strcmp(k,POLARIZATION_)==0)   param = POLARIZATION ; 
    else if(strcmp(k,TIME_)==0)           param = TIME ; 
    else if(strcmp(k,ENERGY_)==0)         param = ENERGY ; 
    else if(strcmp(k,NUMBER_)==0)         param = NUMBER ; 
    return param ;  
}
const char* NGunConfig::getParam(NGunConfig::Param_t par)
{
    const char* p = NULL ; 
    switch(par)
    {
        case COMMENT:      p=COMMENT_      ; break ;
        case PARTICLE:     p=PARTICLE_     ; break ;
        case FRAME:        p=FRAME_        ; break ;
        case POSITION:     p=POSITION_     ; break ;
        case DIRECTION:    p=DIRECTION_    ; break ;
        case POLARIZATION: p=POLARIZATION_ ; break ;
        case TIME:         p=TIME_         ; break ;
        case ENERGY:       p=ENERGY_       ; break ;
        case NUMBER:       p=NUMBER_       ; break ;
        case UNRECOGNIZED: p=UNRECOGNIZED_ ; break ;
    }
    return p ;
}
void NGunConfig::set(Param_t p, const char* s)
{
    switch(p)
    {
        case COMMENT        : setComment(s)        ;break;
        case PARTICLE       : setParticle(s)       ;break;
        case FRAME          : setFrame(s)          ;break;
        case POSITION       : setPositionLocal(s)  ;break;
        case DIRECTION      : setDirectionLocal(s) ;break;
        case POLARIZATION   : setPolarizationLocal(s) ;break;
        case TIME           : setTime(s)           ;break;
        case ENERGY         : setEnergy(s)         ;break;
        case NUMBER         : setNumber(s)         ;break;
        case UNRECOGNIZED   : 
                    LOG(warning) << "NGunConfig::set WARNING ignoring unrecognized parameter " ; 
    }
}


void NGunConfig::init()
{
}

void NGunConfig::parse(std::string config)
{
    if(config.empty())
         parse(NULL);
    else
         parse(config.c_str());
}

void NGunConfig::parse(const char* config_)
{
    m_config = strdup(config_ == NULL ? DEFAULT_CONFIG : config_); 

    std::string config(m_config);
    typedef std::pair<std::string,std::string> KV ; 
    std::vector<KV> ekv = BStr::ekv_split(config.c_str(),'_',"=");

    LOG(debug) << "NGunConfig::parse " <<  config.c_str() ;
    for(std::vector<KV>::const_iterator it=ekv.begin() ; it!=ekv.end() ; it++)
    {
        const char* k = it->first.c_str() ;  
        const char* v = it->second.c_str() ;  
        Param_t p = parseParam(k) ;
        LOG(info) << std::setw(20) << k << ":" << v  ; 

        if(strlen(v)==0)
        {
            LOG(warning) << "NGunConfig::parse skip empty value for key " << k ; 
        }
        else
        {
            set(p, v);
        }
    }
}


void NGunConfig::setComment(const char* s)
{
    m_comment = strdup(s); 
}
void NGunConfig::setParticle(const char* s)
{
    m_particle = strdup(s); 
}
void NGunConfig::setNumber(const char* s)
{
    std::string ss(s);
    m_number = guint_(ss);
}
void NGunConfig::setFrame(const char* s)
{
    std::string ss(s);
    m_frame = givec4(ss);
}

void NGunConfig::setPositionLocal(const char* s)
{
    std::string ss(s);
    glm::vec3 v = gvec3(ss) ; 
    m_position_local.x = v.x;
    m_position_local.y = v.y;
    m_position_local.z = v.z;
    m_position_local.w = 1.0;
}
void NGunConfig::setDirectionLocal(const char* s)
{
    std::string ss(s);
    glm::vec3 v = gvec3(ss) ; 
    m_direction_local.x = v.x;
    m_direction_local.y = v.y;
    m_direction_local.z = v.z;
    m_direction_local.w = 0.0;
}
void NGunConfig::setPolarizationLocal(const char* s)
{
    std::string ss(s);
    glm::vec3 v = gvec3(ss) ; 
    m_polarization_local.x = v.x;
    m_polarization_local.y = v.y;
    m_polarization_local.z = v.z;
    m_polarization_local.w = 0.0;
}





void NGunConfig::setTime(const char* s)
{
    std::string ss(s);
    float time = gfloat_(ss) ;    
    setTime(time);
}
void NGunConfig::setEnergy(const char* s)
{
    std::string ss(s);
    float energy = gfloat_(ss) ;    
    setEnergy(energy);
}




void NGunConfig::setFrameTransform(const char* s)
{
    std::string ss(s);
    bool flip = true ;  
    glm::mat4 transform = gmat4(ss, flip);
    setFrameTransform(transform);
}

void NGunConfig::update()
{
    LOG(info) << "NGunConfig::update" ;
    print(m_frame_transform, "frame_transform");

    glm::vec4 pos = m_frame_transform * m_position_local  ; 
    glm::vec4 dir = m_frame_transform * m_direction_local  ; 
    glm::vec4 pol = m_frame_transform * m_polarization_local  ; 

    setPosition(glm::vec3(pos));
    setDirection(glm::normalize(glm::vec3(dir)));
    setPolarization(glm::normalize(glm::vec3(pol)));
}


void NGunConfig::Summary(const char* msg)
{

    LOG(info) << msg  
              << " comment " << m_comment 
              << " particle " << m_particle
              << " position " << gformat(m_position)
              << " direction " << gformat(m_direction)
              << " polarization " << gformat(m_polarization)
              << " time " << m_time
              << " energy " << m_energy
              << " number " << m_number
              ; 

}



