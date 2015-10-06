#include "TorchStepNPY.hpp"
#include "NPY.hpp"
#include "GLMPrint.hpp"
#include "GLMFormat.hpp"
#include "stringutil.hpp"

#include <vector>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal

const char* TorchStepNPY::DEFAULT_CONFIG = 
    "pos_target=3153;"
    "num_photons=500000;"
    "material_line=102;"
    "direction=0,0,1;"
    "wavelength=500;"
    "weight=1.0;"
    "time=0.1;"
    "zenith_azimuth=0,1,0,1;"
    "radius=0" ;

// NB time 0.f causes 1st step record rendering to be omitted, as zero is special
// TODO: material_line:102 corresponds to GdLS, arrange to detect the material from the pos_target 
 
const char* TorchStepNPY::POS_TARGET_ = "pos_target"; 
const char* TorchStepNPY::POS_OFFSET_ = "pos_offset"; 
const char* TorchStepNPY::DIR_TARGET_ = "dir_target" ; 
const char* TorchStepNPY::NUM_PHOTONS_ = "num_photons" ; 
const char* TorchStepNPY::MATERIAL_LINE_ = "material_line" ; 
const char* TorchStepNPY::DIRECTION_ = "direction" ; 
const char* TorchStepNPY::ZENITH_AZIMUTH_ = "zenith_azimuth" ; 
const char* TorchStepNPY::WAVELENGTH_     = "wavelength" ; 
const char* TorchStepNPY::WEIGHT_     = "weight" ; 
const char* TorchStepNPY::TIME_     = "time" ; 
const char* TorchStepNPY::RADIUS_   = "radius" ; 


TorchStepNPY::Param_t TorchStepNPY::getParam(const char* k)
{
    Param_t param = UNRECOGNIZED ; 
    if(     strcmp(k,POS_TARGET_)==0)     param = POS_TARGET ; 
    else if(strcmp(k,POS_OFFSET_)==0)     param = POS_OFFSET ; 
    else if(strcmp(k,DIR_TARGET_)==0)     param = DIR_TARGET ; 
    else if(strcmp(k,NUM_PHOTONS_)==0)    param = NUM_PHOTONS ; 
    else if(strcmp(k,MATERIAL_LINE_)==0)  param = MATERIAL_LINE ; 
    else if(strcmp(k,DIRECTION_)==0)      param = DIRECTION ; 
    else if(strcmp(k,ZENITH_AZIMUTH_)==0) param = ZENITH_AZIMUTH ; 
    else if(strcmp(k,WAVELENGTH_)==0)     param = WAVELENGTH ; 
    else if(strcmp(k,WEIGHT_)==0)         param = WEIGHT ; 
    else if(strcmp(k,TIME_)==0)           param = TIME ; 
    else if(strcmp(k,RADIUS_)==0)         param = RADIUS ; 
    return param ;  
}

void TorchStepNPY::configure(const char* config_)
{
    std::string config(config_);
    typedef std::pair<std::string,std::string> KV ; 
    std::vector<KV> ekv = ekv_split(config.c_str(),';',"=");

    printf("TorchStepNPY::configure %s \n", config.c_str() );
    for(std::vector<KV>::const_iterator it=ekv.begin() ; it!=ekv.end() ; it++)
    {
        printf(" %20s : %s \n", it->first.c_str(), it->second.c_str() );
        set(getParam(it->first.c_str()), it->second.c_str());
    }
    setGenstepId();
}

void TorchStepNPY::set(Param_t p, const char* s)
{
    switch(p)
    {
        case POS_TARGET     : setPosTarget(s)      ;break;
        case POS_OFFSET     : setPosOffset(s)      ;break;
        case DIR_TARGET     : setDirTarget(s)      ;break;
        case NUM_PHOTONS    : setNumPhotons(s)     ;break;
        case MATERIAL_LINE  : setMaterialLine(s)   ;break;
        case DIRECTION      : setDirection(s)      ;break;
        case ZENITH_AZIMUTH : setZenithAzimuth(s)  ;break;
        case WAVELENGTH     : setWavelength(s)     ;break;
        case WEIGHT         : setWeight(s)         ;break;
        case TIME           : setTime(s)           ;break;
        case RADIUS         : setRadius(s)         ;break;
        case UNRECOGNIZED   : 
                    LOG(warning) << "TorchStepNPY::set WARNING ignoring unrecognized parameter " ; 
    }
}


void TorchStepNPY::setPosTarget(const char* s)
{
    std::string ss(s);
    m_pos_target = givec4(ss);
}
void TorchStepNPY::setPosTarget(unsigned int index, unsigned int mesh)
{
    m_pos_target.x = index ; 
    m_pos_target.y = mesh ; 
}


void TorchStepNPY::setPosOffset(const char* s)
{
    std::string ss(s);
    m_pos_offset = gvec3(ss) ;
}




void TorchStepNPY::setDirTarget(const char* s)
{
    std::string ss(s);
    m_dir_target = givec4(ss) ;
}
void TorchStepNPY::setDirTarget(unsigned int index, unsigned int mesh)
{
    m_dir_target.x = index ; 
    m_dir_target.y = mesh ; 
}




void TorchStepNPY::setNumPhotons(const char* s)
{
    setNumPhotons(boost::lexical_cast<unsigned int>(s)) ; 
}
void TorchStepNPY::setNumPhotons(unsigned int num_photons)
{
    m_ctrl.w = num_photons ; 
}




void TorchStepNPY::setMaterialLine(const char* s)
{
    m_ctrl.z = boost::lexical_cast<int>(s) ; 
}
void TorchStepNPY::setDirection(const char* s)
{
    std::string ss(s);
    glm::vec3 dir = gvec3(ss) ;
    m_dirw.x = dir.x ; 
    m_dirw.y = dir.y ; 
    m_dirw.z = dir.z ; 
}
void TorchStepNPY::setWavelength(const char* s)
{
    m_polw.w = boost::lexical_cast<float>(s) ;
}


void TorchStepNPY::setWeight(const char* s)
{
    m_dirw.w = boost::lexical_cast<float>(s) ;
}
void TorchStepNPY::setTime(const char* s)
{
    m_post.w = boost::lexical_cast<float>(s) ;
}


void TorchStepNPY::setRadius(const char* s)
{
    setRadius(boost::lexical_cast<float>(s)) ;
}
void TorchStepNPY::setRadius(float radius)
{
    m_beam.x = radius ;
}






void TorchStepNPY::setZenithAzimuth(const char* s)
{
    std::string ss(s);
    m_zenith_azimuth = gvec4(ss) ;
}

NPY<float>* TorchStepNPY::getNPY()
{
    assert( m_step_index == m_num_step ); // TorchStepNPY is incomplete
    return m_npy ; 
}

void TorchStepNPY::addStep()
{
    if(m_npy == NULL)
    {
        m_npy = NPY<float>::make(m_num_step, 6, 4);
        m_npy->zero();
    }
   
    unsigned int i = m_step_index ; 

    m_npy->setQuadI(i, 0, m_ctrl );
    m_npy->setQuad( i, 1, m_post );
    m_npy->setQuad( i, 2, m_dirw );
    m_npy->setQuad( i, 3, m_polw );
    m_npy->setQuad( i, 4, m_zenith_azimuth );
    m_npy->setQuad( i, 5, m_beam );

    m_step_index++ ; 
}

void TorchStepNPY::dump(const char* msg)
{
    printf("%s config %s  \n", msg, m_config );

    print(m_pos_target, "m_pos_target ");
    print(m_dir_target, "m_dir_target ");

    print(m_ctrl, "m_ctrl : id/pid/MaterialLine/NumPhotons" );
    print(m_post, "m_post : position, time " ); 
    print(m_dirw, "m_dirw : direction, weight" ); 
    print(m_polw, "m_polw : polarization, wavelength" ); 
    print(m_zenith_azimuth, "m_zenith_azimuth : zenith, azimuth " ); 
    print(m_beam, "m_beam: radius,... " ); 
}


