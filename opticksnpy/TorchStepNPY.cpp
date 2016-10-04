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
#include "GLMPrint.hpp"
#include "GLMFormat.hpp"
#include "uif.h"
#include "GenstepNPY.hpp"
#include "TorchStepNPY.hpp"
#include "PLOG.hh"

const char* TorchStepNPY::DEFAULT_CONFIG = 
    "type=sphere_"
    "frame=3153_"
    "source=0,0,0_"
    "target=0,0,1_"
    "photons=100000_"
    "material=GdDopedLS_"   
    "wavelength=430_"
    "weight=1.0_"
    "time=0.1_"
    "zenithazimuth=0,1,0,1_"
    "radius=0_" ;

//  Aug 2016: change default torch wavelength from 380nm to 430nm
//
//
// NB time 0.f causes 1st step record rendering to be omitted, as zero is special
// NB the material string needs to be externally translated into a material line
//
// TODO: material lookup based on the frame volume solid ?
//
 
const char* TorchStepNPY::T_UNDEF_    = "undef" ; 
const char* TorchStepNPY::T_POINT_    = "point" ; 
const char* TorchStepNPY::T_SPHERE_    = "sphere" ; 
const char* TorchStepNPY::T_DISC_      = "disc" ; 
const char* TorchStepNPY::T_DISC_INTERSECT_SPHERE_ = "discIntersectSphere" ; 
const char* TorchStepNPY::T_DISC_INTERSECT_SPHERE_DUMB_ = "discIntersectSphereDumb" ; 
const char* TorchStepNPY::T_DISCLIN_   = "disclin" ; 
const char* TorchStepNPY::T_DISCAXIAL_   = "discaxial" ; 
const char* TorchStepNPY::T_INVSPHERE_ = "invsphere" ; 
const char* TorchStepNPY::T_REFLTEST_  = "refltest" ; 
const char* TorchStepNPY::T_INVCYLINDER_ = "invcylinder" ; 
const char* TorchStepNPY::T_RING_ = "ring" ; 

Torch_t TorchStepNPY::parseType(const char* k)
{
    Torch_t type = T_UNDEF ;
    if(       strcmp(k,T_SPHERE_)==0)    type = T_SPHERE ; 
    else if(  strcmp(k,T_POINT_)==0)     type = T_POINT ; 
    else if(  strcmp(k,T_DISC_)==0)      type = T_DISC ; 
    else if(  strcmp(k,T_DISCLIN_)==0)   type = T_DISCLIN ; 
    else if(  strcmp(k,T_DISCAXIAL_)==0) type = T_DISCAXIAL ; 
    else if(  strcmp(k,T_DISC_INTERSECT_SPHERE_)==0) type = T_DISC_INTERSECT_SPHERE ; 
    else if(  strcmp(k,T_DISC_INTERSECT_SPHERE_DUMB_)==0) type = T_DISC_INTERSECT_SPHERE_DUMB ; 
    else if(  strcmp(k,T_INVSPHERE_)==0) type = T_INVSPHERE ; 
    else if(  strcmp(k,T_INVCYLINDER_)==0) type = T_INVCYLINDER ; 
    else if(  strcmp(k,T_REFLTEST_)==0)  type = T_REFLTEST ; 
    else if(  strcmp(k,T_RING_)==0)      type = T_RING ; 
    return type ;   
}





::Torch_t TorchStepNPY::getType()
{
    unsigned utype = getBaseType();
    Torch_t type = T_UNDEF ;
    switch(utype)
    {
       case T_SPHERE:      type=T_SPHERE     ;break; 
       case T_DISC:        type=T_DISC       ;break; 
       case T_POINT:       type=T_POINT      ;break; 
       case T_DISCLIN:     type=T_DISCLIN    ;break; 
       case T_DISCAXIAL:   type=T_DISCAXIAL  ;break; 
       case T_DISC_INTERSECT_SPHERE:   type=T_DISC_INTERSECT_SPHERE  ;break; 
       case T_DISC_INTERSECT_SPHERE_DUMB:   type=T_DISC_INTERSECT_SPHERE_DUMB  ;break; 
       case T_INVSPHERE:   type=T_INVSPHERE  ;break; 
       case T_REFLTEST:    type=T_REFLTEST   ;break; 
       case T_INVCYLINDER: type=T_INVCYLINDER;break; 
       case T_RING:        type=T_RING       ;break; 
    }
    return type ; 
}

const char* TorchStepNPY::getTypeName()
{
    ::Torch_t type = getType();
    const char* name = NULL ; 
    switch(type)
    {
       case T_SPHERE:      name=T_SPHERE_     ;break; 
       case T_DISC:        name=T_DISC_       ;break; 
       case T_POINT:       name=T_POINT_      ;break; 
       case T_DISCLIN:     name=T_DISCLIN_    ;break; 
       case T_DISCAXIAL:   name=T_DISCAXIAL_  ;break; 
       case T_DISC_INTERSECT_SPHERE:   name=T_DISC_INTERSECT_SPHERE_  ;break; 
       case T_DISC_INTERSECT_SPHERE_DUMB:   name=T_DISC_INTERSECT_SPHERE_DUMB_  ;break; 
       case T_INVSPHERE:   name=T_INVSPHERE_  ;break; 
       case T_REFLTEST:    name=T_REFLTEST_   ;break; 
       case T_INVCYLINDER: name=T_INVCYLINDER_;break; 
       case T_RING:        name=T_RING_       ;break; 

       case T_UNDEF:       name=T_UNDEF_      ;break; 
       default:    
                           assert(0)  ;

    }
    return name ; 
}


void TorchStepNPY::setType(const char* s)
{
    ::Torch_t type = parseType(s) ;
    unsigned utype = unsigned(type);
    setBaseType(utype);
}





const char* TorchStepNPY::M_SPOL_ = "spol" ; 
const char* TorchStepNPY::M_PPOL_ = "ppol" ; 
const char* TorchStepNPY::M_FIXPOL_ = "fixpol" ; 
const char* TorchStepNPY::M_FLAT_THETA_ = "flatTheta" ; 
const char* TorchStepNPY::M_FLAT_COSTHETA_ = "flatCosTheta" ; 
const char* TorchStepNPY::M_WAVELENGTH_SOURCE_ = "wavelengthSource" ; 
const char* TorchStepNPY::M_WAVELENGTH_COMB_ = "wavelengthComb" ; 

Mode_t TorchStepNPY::parseMode(const char* k)
{
    Mode_t mode = M_UNDEF ;
    if(       strcmp(k,M_SPOL_)==0)      mode = M_SPOL ; 
    else if(  strcmp(k,M_PPOL_)==0)      mode = M_PPOL ; 
    else if(  strcmp(k,M_FLAT_THETA_)==0)      mode = M_FLAT_THETA ; 
    else if(  strcmp(k,M_FLAT_COSTHETA_)==0)   mode = M_FLAT_COSTHETA ; 
    else if(  strcmp(k,M_FIXPOL_)==0)          mode = M_FIXPOL ; 
    else if(  strcmp(k,M_WAVELENGTH_SOURCE_)==0)  mode = M_WAVELENGTH_SOURCE ; 
    else if(  strcmp(k,M_WAVELENGTH_COMB_)==0)  mode = M_WAVELENGTH_COMB ; 
    return mode ;   
}

::Mode_t TorchStepNPY::getMode()
{
    ::Mode_t mode = (::Mode_t)getBaseMode() ;
    return mode ; 
}



std::string TorchStepNPY::getModeString()
{
    std::stringstream ss ; 
    ::Mode_t mode = getMode();

    if(mode & M_SPOL) ss << M_SPOL_ << " " ;
    if(mode & M_PPOL) ss << M_PPOL_ << " " ;
    if(mode & M_FLAT_THETA) ss << M_FLAT_THETA_ << " " ; 
    if(mode & M_FLAT_COSTHETA) ss << M_FLAT_COSTHETA_ << " " ; 
    if(mode & M_FIXPOL) ss << M_FIXPOL_ << " " ; 
    if(mode & M_WAVELENGTH_SOURCE) ss << M_WAVELENGTH_SOURCE_ << " " ; 
    if(mode & M_WAVELENGTH_COMB) ss << M_WAVELENGTH_COMB_ << " " ; 

    return ss.str();
} 



std::string TorchStepNPY::description()
{
    glm::vec3 pos = getPosition() ;
    glm::vec3 dir = getDirection() ;
    glm::vec3 pol = getPolarization() ;

    std::stringstream ss ; 
    ss
        << " typeName " << getTypeName() 
        << " modeString " << getModeString() 
        << " position " << gformat(pos)
        << " direction " << gformat(dir)
        << " polarization " << gformat(pol)
        << " radius " << getRadius()
        << " wavelength " << getWavelength()
        << " time " << getTime()
        ; 

    return ss.str();
}






const char* TorchStepNPY::TYPE_ = "type"; 
const char* TorchStepNPY::MODE_ = "mode"; 
const char* TorchStepNPY::POLARIZATION_ = "polarization"; 
const char* TorchStepNPY::FRAME_ = "frame"; 
const char* TorchStepNPY::TRANSFORM_ = "transform"; 
const char* TorchStepNPY::SOURCE_ = "source"; 
const char* TorchStepNPY::TARGET_ = "target" ; 
const char* TorchStepNPY::PHOTONS_ = "photons" ; 
const char* TorchStepNPY::MATERIAL_ = "material" ; 
const char* TorchStepNPY::ZENITHAZIMUTH_ = "zenithazimuth" ; 
const char* TorchStepNPY::WAVELENGTH_     = "wavelength" ; 
const char* TorchStepNPY::WEIGHT_     = "weight" ; 
const char* TorchStepNPY::TIME_     = "time" ; 
const char* TorchStepNPY::RADIUS_   = "radius" ; 
const char* TorchStepNPY::DISTANCE_   = "distance" ; 

TorchStepNPY::Param_t TorchStepNPY::parseParam(const char* k)
{
    Param_t param = UNRECOGNIZED ; 
    if(     strcmp(k,FRAME_)==0)          param = FRAME ; 
    else if(strcmp(k,TRANSFORM_)==0)      param = TRANSFORM ; 
    else if(strcmp(k,TYPE_)==0)           param = TYPE ; 
    else if(strcmp(k,MODE_)==0)           param = MODE ; 
    else if(strcmp(k,POLARIZATION_)==0)   param = POLARIZATION ; 
    else if(strcmp(k,SOURCE_)==0)         param = SOURCE ; 
    else if(strcmp(k,TARGET_)==0)         param = TARGET ; 
    else if(strcmp(k,PHOTONS_)==0)        param = PHOTONS ; 
    else if(strcmp(k,MATERIAL_)==0)       param = MATERIAL ; 
    else if(strcmp(k,ZENITHAZIMUTH_)==0)  param = ZENITHAZIMUTH ; 
    else if(strcmp(k,WAVELENGTH_)==0)     param = WAVELENGTH ; 
    else if(strcmp(k,WEIGHT_)==0)         param = WEIGHT ; 
    else if(strcmp(k,TIME_)==0)           param = TIME ; 
    else if(strcmp(k,RADIUS_)==0)         param = RADIUS ; 
    else if(strcmp(k,DISTANCE_)==0)       param = DISTANCE ; 
    return param ;  
}

void TorchStepNPY::set(Param_t p, const char* s)
{
    bool ignore = false ; 
    switch(p)
    {
        case TYPE           : setType(s)           ;break;
        case MODE           : setMode(s)           ;break;
        case TRANSFORM      : setFrameTransform(s) ;break;
        case FRAME          : setFrame(s)          ;break;
        case POLARIZATION   : setPolarizationLocal(s) ;break;
        case SOURCE         : setSourceLocal(s)    ;break;
        case TARGET         : setTargetLocal(s)    ;break;
        case PHOTONS        : setNumPhotons(s)     ;break;
        case MATERIAL       : setMaterial(s)       ;break;
        case ZENITHAZIMUTH  : setZenithAzimuth(s)  ;break;
        case WAVELENGTH     : setWavelength(s)     ;break;
        case WEIGHT         : setWeight(s)         ;break;
        case TIME           : setTime(s)           ;break;
        case RADIUS         : setRadius(s)         ;break;
        case DISTANCE       : setDistance(s)       ;break;
        case UNRECOGNIZED   : ignore=true ;  
    }

    if(ignore)
    {
        LOG(fatal) << "TorchStepNPY::set WARNING ignoring unrecognized parameter [" << s << "]"  ; 
        assert(0);
    }
}




TorchStepNPY::TorchStepNPY(unsigned genstep_type, unsigned int num_step, const char* config) 
       :  
       GenstepNPY(genstep_type,  num_step, config ? strdup(config) : DEFAULT_CONFIG ),
       m_num_photons_per_g4event(10000)
{
   init();
}


void TorchStepNPY::init()
{
    const char* config = getConfig(); 

    typedef std::pair<std::string,std::string> KV ; 
    std::vector<KV> ekv = BStr::ekv_split(config,'_',"=");

    LOG(debug) << "TorchStepNPY::init " <<  config ;
    for(std::vector<KV>::const_iterator it=ekv.begin() ; it!=ekv.end() ; it++)
    {
        const char* k = it->first.c_str() ;  
        const char* v = it->second.c_str() ;  

        Param_t p = parseParam(k) ;

        LOG(debug) << std::setw(20) << k << ":" << v  ; 

        if(p == UNRECOGNIZED)
        {
            LOG(fatal) << "TorchStepNPY::init "
                       << " UNRECOGNIZED parameter "
                       << " [" << k << "] = [" << v << "]"
                       ;
            assert(0);
        }


        if(strlen(v)==0)
        {
            LOG(warning) << "TorchStepNPY::init skip empty value for key " << k ; 
        }
        else
        {
            set(p, v);
        }
    }
}





glm::vec4& TorchStepNPY::getSourceLocal()
{
    return m_source_local ; 
}
glm::vec4& TorchStepNPY::getTargetLocal()
{
    return m_target_local ; 
}
glm::vec4& TorchStepNPY::getPolarizationLocal()
{
    return m_polarization_local ; 
}





// used from cfg4-
void TorchStepNPY::setNumPhotonsPerG4Event(unsigned int n)
{
    m_num_photons_per_g4event = n ; 
}
unsigned int TorchStepNPY::getNumPhotonsPerG4Event()
{
    return m_num_photons_per_g4event ;
}
unsigned int TorchStepNPY::getNumG4Event()
{
    unsigned int num_photons = getNumPhotons();
    unsigned int ppe = m_num_photons_per_g4event ; 
    unsigned int num_g4event ; 
    if(num_photons < ppe)
    {
        num_g4event = 1 ; 
    }
    else
    {
        assert( num_photons % ppe == 0 && "expecting num_photons to be exactly divisible by NumPhotonsPerG4Event " );
        num_g4event = num_photons / ppe ; 
    }
    return num_g4event ; 
}




bool TorchStepNPY::isIncidentSphere()
{
    ::Torch_t type = getType();
    return type == T_DISC_INTERSECT_SPHERE  ;
}

bool TorchStepNPY::isDisc()
{
    ::Torch_t type = getType();
    return type == T_DISC  ;
}
bool TorchStepNPY::isDiscLinear()
{
    ::Torch_t type = getType();
    return type == T_DISCLIN  ;
}
bool TorchStepNPY::isRing()
{
    ::Torch_t type = getType();
    return type == T_RING  ;
}
bool TorchStepNPY::isPoint()
{
    ::Torch_t type = getType();
    return type == T_POINT  ;
}
bool TorchStepNPY::isSphere()
{
    ::Torch_t type = getType();
    return type == T_SPHERE  ;
}

bool TorchStepNPY::isReflTest()
{
    ::Torch_t type = getType();
    return type == T_REFLTEST ;
}


bool TorchStepNPY::isSPolarized()
{
    ::Mode_t  mode = getMode();
    return (mode & M_SPOL) != 0  ;
}
bool TorchStepNPY::isPPolarized()
{
    ::Mode_t  mode = getMode();
    return (mode & M_PPOL) != 0  ;
}
bool TorchStepNPY::isFixPolarized()
{
    ::Mode_t  mode = getMode();
    return (mode & M_FIXPOL) != 0  ;
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

void TorchStepNPY::setPolarizationLocal(const char* s)
{
    std::string ss(s);
    glm::vec3 v = gvec3(ss) ; 
    m_polarization_local.x = v.x;
    m_polarization_local.y = v.y;
    m_polarization_local.z = v.z;
    m_polarization_local.w = 0.0;  // direction not position
}





void TorchStepNPY::update()
{
   // direction from: target - source
   // position from : source

    const glm::mat4& frame_transform = getFrameTransform() ;

    m_src = frame_transform * m_source_local  ; 
    m_tgt = frame_transform * m_target_local  ; 
    m_pol = frame_transform * m_polarization_local  ; 

    m_dir = glm::vec3(m_tgt) - glm::vec3(m_src) ;

    glm::vec3 dir = glm::normalize( m_dir );

    setPosition(m_src);
    setDirection(dir);
    setPolarization(m_pol);
}


void TorchStepNPY::setMode(const char* s)
{
    std::vector<std::string> tp; 
    boost::split(tp, s, boost::is_any_of(","));

    //::Mode_t mode = M_UNDEF ; 
    unsigned int mode = 0 ; 

    for(unsigned int i=0 ; i < tp.size() ; i++)
    {
        const char* emode_ = tp[i].c_str() ;
        ::Mode_t emode = parseMode(emode_) ;
        mode |= emode ; 
        LOG(debug) << "TorchStepNPY::setMode" 
                  << " emode_ " << emode_
                  << " emode " << emode
                  << " mode " << mode 
                  ; 
    }

    setBaseMode(mode);
}



void TorchStepNPY::dump(const char* msg)
{
    LOG(info) << msg  ;

    print(m_source_local,       "m_source_local       ", m_src, "m_src");
    print(m_target_local,       "m_target_local       ", m_tgt, "m_tgt");
    print(m_polarization_local, "m_polarization_local ", m_pol, "m_pol");

    print(m_dir, "m_dir [normalize(m_tgt - m_src)] ");

    GenstepNPY::dumpBase(msg);
}



void TorchStepNPY::Summary(const char* msg)
{
    LOG(info) << msg  << description() ; 
}






