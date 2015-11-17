#include "TorchStepNPY.hpp"
#include "NPY.hpp"
#include "GLMPrint.hpp"
#include "GLMFormat.hpp"
#include "stringutil.hpp"
#include "uif.h"

#include <vector>
#include <iomanip>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal

const char* TorchStepNPY::DEFAULT_CONFIG = 
    "type=sphere_"
    "frame=3153_"
    "source=0,0,0_"
    "target=0,0,1_"
    "photons=500000_"
    "material=GdDopedLS_"   
    "wavelength=500_"
    "weight=1.0_"
    "time=0.1_"
    "zenithazimuth=0,1,0,1_"
    "radius=0_" ;

// NB time 0.f causes 1st step record rendering to be omitted, as zero is special
// NB the material string needs to be externally translated into a material line
//
// TODO: material lookup based on the frame volume solid ?
//
 
const char* TorchStepNPY::T_SPHERE_ = "sphere" ; 
const char* TorchStepNPY::T_DISC_   = "disc" ; 
const char* TorchStepNPY::T_INVSPHERE_ = "invsphere" ; 
const char* TorchStepNPY::T_REFLTEST_ = "refltest" ; 

Torch_t TorchStepNPY::parseType(const char* k)
{
    Torch_t type = T_UNDEF ;
    if(       strcmp(k,T_SPHERE_)==0)    type = T_SPHERE ; 
    else if(  strcmp(k,T_DISC_)==0)      type = T_DISC ; 
    else if(  strcmp(k,T_INVSPHERE_)==0) type = T_INVSPHERE ; 
    else if(  strcmp(k,T_REFLTEST_)==0)  type = T_REFLTEST ; 
    return type ;   
}

void TorchStepNPY::setType(const char* s)
{
    ::Torch_t type = parseType(s) ;
    uif_t uif ; 
    uif.u = type ; 
    m_beam.w = uif.f ;
}

::Torch_t TorchStepNPY::getType()
{
    uif_t uif ;
    uif.f = m_beam.w ; 
    Torch_t type = T_UNDEF ;
    switch(uif.u)
    {
       case T_SPHERE:      type=T_SPHERE     ;break; 
       case T_DISC:        type=T_DISC       ;break; 
       case T_INVSPHERE:   type=T_INVSPHERE  ;break; 
       case T_REFLTEST:    type=T_REFLTEST   ;break; 
    }
    return type ; 
}



const char* TorchStepNPY::M_SPOL_ = "spol" ; 
const char* TorchStepNPY::M_PPOL_ = "ppol" ; 

Polz_t TorchStepNPY::parsePolz(const char* k)
{
    Polz_t mode = M_UNDEF ;
    if(       strcmp(k,M_SPOL_)==0)      mode = M_SPOL ; 
    else if(  strcmp(k,M_PPOL_)==0)      mode = M_PPOL ; 
    return mode ;   
}

::Polz_t TorchStepNPY::getPolz()
{
    uif_t uif ;
    uif.f = m_beam.z ; 
    Polz_t mode = M_UNDEF ;
    switch(uif.u)
    {
       case M_SPOL: mode=M_SPOL ;break; 
       case M_PPOL: mode=M_PPOL ;break; 
    }
    return mode ; 
}

void TorchStepNPY::setPolz(const char* s)
{
    ::Polz_t mode = parsePolz(s) ;
    uif_t uif ; 
    uif.u = mode ; 
    m_beam.z = uif.f ;
}

const char* TorchStepNPY::TYPE_ = "type"; 
const char* TorchStepNPY::POLZ_ = "polz"; 
const char* TorchStepNPY::FRAME_ = "frame"; 
const char* TorchStepNPY::SOURCE_ = "source"; 
const char* TorchStepNPY::TARGET_ = "target" ; 
const char* TorchStepNPY::PHOTONS_ = "photons" ; 
const char* TorchStepNPY::MATERIAL_ = "material" ; 
const char* TorchStepNPY::ZENITHAZIMUTH_ = "zenithazimuth" ; 
const char* TorchStepNPY::WAVELENGTH_     = "wavelength" ; 
const char* TorchStepNPY::WEIGHT_     = "weight" ; 
const char* TorchStepNPY::TIME_     = "time" ; 
const char* TorchStepNPY::RADIUS_   = "radius" ; 

TorchStepNPY::Param_t TorchStepNPY::parseParam(const char* k)
{
    Param_t param = UNRECOGNIZED ; 
    if(     strcmp(k,FRAME_)==0)          param = FRAME ; 
    else if(strcmp(k,TYPE_)==0)           param = TYPE ; 
    else if(strcmp(k,POLZ_)==0)           param = POLZ ; 
    else if(strcmp(k,SOURCE_)==0)         param = SOURCE ; 
    else if(strcmp(k,TARGET_)==0)         param = TARGET ; 
    else if(strcmp(k,PHOTONS_)==0)        param = PHOTONS ; 
    else if(strcmp(k,MATERIAL_)==0)       param = MATERIAL ; 
    else if(strcmp(k,ZENITHAZIMUTH_)==0)  param = ZENITHAZIMUTH ; 
    else if(strcmp(k,WAVELENGTH_)==0)     param = WAVELENGTH ; 
    else if(strcmp(k,WEIGHT_)==0)         param = WEIGHT ; 
    else if(strcmp(k,TIME_)==0)           param = TIME ; 
    else if(strcmp(k,RADIUS_)==0)         param = RADIUS ; 
    return param ;  
}

void TorchStepNPY::set(Param_t p, const char* s)
{
    switch(p)
    {
        case TYPE           : setType(s)           ;break;
        case POLZ           : setPolz(s)           ;break;
        case FRAME          : setFrame(s)          ;break;
        case SOURCE         : setSourceLocal(s)    ;break;
        case TARGET         : setTargetLocal(s)    ;break;
        case PHOTONS        : setNumPhotons(s)     ;break;
        case MATERIAL       : setMaterial(s)       ;break;
        case ZENITHAZIMUTH  : setZenithAzimuth(s)  ;break;
        case WAVELENGTH     : setWavelength(s)     ;break;
        case WEIGHT         : setWeight(s)         ;break;
        case TIME           : setTime(s)           ;break;
        case RADIUS         : setRadius(s)         ;break;
        case UNRECOGNIZED   : 
                    LOG(warning) << "TorchStepNPY::set WARNING ignoring unrecognized parameter " ; 
    }
}


void TorchStepNPY::configure(const char* config_)
{
    m_config = strdup(config_); 

    std::string config(config_);
    typedef std::pair<std::string,std::string> KV ; 
    std::vector<KV> ekv = ekv_split(config.c_str(),'_',"=");

    LOG(debug) << "TorchStepNPY::configure " <<  config.c_str() ;
    for(std::vector<KV>::const_iterator it=ekv.begin() ; it!=ekv.end() ; it++)
    {
        LOG(debug) << std::setw(20) << it->first << ":" << it->second ; 
        set(parseParam(it->first.c_str()), it->second.c_str());
    }
    setGenstepId();
}

void TorchStepNPY::setFrame(const char* s)
{
    std::string ss(s);
    m_frame = givec4(ss);
}
void TorchStepNPY::setFrame(unsigned int vindex)
{
    m_frame.x = vindex ; 
    m_frame.y = 0 ; 
    m_frame.z = 0 ; 
    m_frame.w = 0 ; 
}

void TorchStepNPY::setSourceLocal(const char* s)
{
    std::string ss(s);
    glm::vec3 v = gvec3(ss) ; 
    m_source_local.x = v.x;
    m_source_local.y = v.y;
    m_source_local.z = v.z;
    m_source_local.w = 1.0;
}
void TorchStepNPY::setTargetLocal(const char* s)
{
    std::string ss(s);
    glm::vec3 v = gvec3(ss) ; 
    m_target_local.x = v.x;
    m_target_local.y = v.y;
    m_target_local.z = v.z;
    m_target_local.w = 1.0;
}

void TorchStepNPY::setNumPhotons(const char* s)
{
    setNumPhotons(boost::lexical_cast<unsigned int>(s)) ; 
}
void TorchStepNPY::setNumPhotons(unsigned int num_photons)
{
    m_ctrl.w = num_photons ; 
}
void TorchStepNPY::setMaterial(const char* s)
{
    m_material = strdup(s);
}
void TorchStepNPY::setMaterialLine(unsigned int ml)
{
    m_ctrl.z = ml ; 
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
void TorchStepNPY::addStep(bool verbose)
{
    if(m_npy == NULL)
    {
        m_npy = NPY<float>::make(m_num_step, 6, 4);
        m_npy->zero();
    }
   
    unsigned int i = m_step_index ; 

    update(); 

    if(verbose) dump("TorchStepNPY::addStep");

    m_npy->setQuadI(i, 0, m_ctrl );
    m_npy->setQuad( i, 1, m_post );
    m_npy->setQuad( i, 2, m_dirw );
    m_npy->setQuad( i, 3, m_polw );
    m_npy->setQuad( i, 4, m_zenith_azimuth );
    m_npy->setQuad( i, 5, m_beam );

    m_step_index++ ; 
}

void TorchStepNPY::update()
{
    m_src = m_frame_transform * m_source_local  ; 
    m_tgt = m_frame_transform * m_target_local  ; 
    m_dir = glm::vec3(m_tgt) - glm::vec3(m_src) ;

    m_post.x = m_src.x ; 
    m_post.y = m_src.y ; 
    m_post.z = m_src.z ; 

    glm::vec3 dir = glm::normalize( m_dir );

    m_dirw.x = dir.x ; 
    m_dirw.y = dir.y ; 
    m_dirw.z = dir.z ; 
}


void TorchStepNPY::dump(const char* msg)
{
    LOG(info) << msg  
              << " config " << m_config 
              << " material " << m_material
              ; 

    print(m_frame,        "m_frame ");
    print(m_source_local, "m_source_local ", m_src, "m_src");
    print(m_target_local, "m_target_local ", m_tgt, "m_tgt");
    print(m_dir, "m_die [normalize(m_tgt - m_src)] ");


    //print(m_ctrl, "m_ctrl : id/pid/MaterialLine/NumPhotons" );
    print(m_post, "m_post : position, time " ); 
    print(m_dirw, "m_dirw : direction, weight" ); 
    //print(m_polw, "m_polw : polarization, wavelength" ); 
    //print(m_zenith_azimuth, "m_zenith_azimuth : zenith, azimuth " ); 
    //print(m_beam, "m_beam: radius,... " ); 
}

