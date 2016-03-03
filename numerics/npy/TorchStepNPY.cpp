#include "TorchStepNPY.hpp"
#include "NPY.hpp"
#include "GLMPrint.hpp"
#include "GLMFormat.hpp"
#include "stringutil.hpp"
#include "uif.h"

#include <vector>
#include <iomanip>
#include <sstream>

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
    "wavelength=380_"
    "weight=1.0_"
    "time=0.1_"
    "zenithazimuth=0,1,0,1_"
    "radius=0_" ;

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
    uif_t uif ;
    uif.f = m_beam.w ; 
    Torch_t type = T_UNDEF ;
    switch(uif.u)
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
    uif_t uif ; 
    uif.u = type ; 
    m_beam.w = uif.f ;
}




const char* TorchStepNPY::M_SPOL_ = "spol" ; 
const char* TorchStepNPY::M_PPOL_ = "ppol" ; 
const char* TorchStepNPY::M_FLAT_THETA_ = "flatTheta" ; 
const char* TorchStepNPY::M_FLAT_COSTHETA_ = "flatCosTheta" ; 

Mode_t TorchStepNPY::parseMode(const char* k)
{
    Mode_t mode = M_UNDEF ;
    if(       strcmp(k,M_SPOL_)==0)      mode = M_SPOL ; 
    else if(  strcmp(k,M_PPOL_)==0)      mode = M_PPOL ; 
    else if(  strcmp(k,M_FLAT_THETA_)==0)      mode = M_FLAT_THETA ; 
    else if(  strcmp(k,M_FLAT_COSTHETA_)==0)   mode = M_FLAT_COSTHETA ; 
    return mode ;   
}

::Mode_t TorchStepNPY::getMode()
{
    uif_t uif ;
    uif.f = m_beam.z ; 

    ::Mode_t mode = (::Mode_t)uif.u ;

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

void TorchStepNPY::setFrameTransform(const char* s)
{
    std::string ss(s);
    bool flip = true ;  
    glm::mat4 transform = gmat4(ss, flip);
    setFrameTransform(transform);
    setFrameTargetted(true);
}

void TorchStepNPY::setMaterial(const char* s)
{
    m_material = strdup(s);
}

void TorchStepNPY::setDirection(const char* s)
{
    std::string ss(s);
    glm::vec3 dir = gvec3(ss) ;
    setDirection(dir);
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

    m_src = m_frame_transform * m_source_local  ; 
    m_tgt = m_frame_transform * m_target_local  ; 
    m_pol = m_frame_transform * m_polarization_local  ; 

    m_dir = glm::vec3(m_tgt) - glm::vec3(m_src) ;

    glm::vec3 dir = glm::normalize( m_dir );

    setPosition(m_src);
    setDirection(dir);
    setPolarization(m_pol);
}




/// seqializing and setting the transport quads

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

    m_npy->setQuadI(m_ctrl, i, 0 );
    m_npy->setQuad( m_post, i, 1);
    m_npy->setQuad( m_dirw, i, 2);
    m_npy->setQuad( m_polw, i, 3);
    m_npy->setQuad( m_zeaz, i, 4);
    m_npy->setQuad( m_beam, i, 5);

    m_step_index++ ; 
}

NPY<float>* TorchStepNPY::getNPY()
{
    assert( m_step_index == m_num_step ); // TorchStepNPY is incomplete
    return m_npy ; 
}




// m_ctrl

void TorchStepNPY::setGenstepId()
{
   m_ctrl.x = m_genstep_id ; 
}


void TorchStepNPY::setMaterialLine(unsigned int ml)
{
    m_ctrl.z = ml ; 
}

void TorchStepNPY::setNumPhotons(const char* s)
{
    setNumPhotons(boost::lexical_cast<unsigned int>(s)) ; 
}
void TorchStepNPY::setNumPhotons(unsigned int num_photons)
{
    m_ctrl.w = num_photons ; 
}
unsigned int TorchStepNPY::getNumPhotons()
{
    return m_ctrl.w ; 
}




// m_post

void TorchStepNPY::setPosition(const glm::vec4& pos)
{
    m_post.x = pos.x ; 
    m_post.y = pos.y ; 
    m_post.z = pos.z ; 
}

void TorchStepNPY::setTime(const char* s)
{
    m_post.w = boost::lexical_cast<float>(s) ;
}
float TorchStepNPY::getTime()
{
    return m_post.w ; 
}

glm::vec3 TorchStepNPY::getPosition()
{
    return glm::vec3(m_post);
}




// m_dirw

void TorchStepNPY::setDirection(const glm::vec3& dir)
{
    m_dirw.x = dir.x ; 
    m_dirw.y = dir.y ; 
    m_dirw.z = dir.z ; 
}

glm::vec3 TorchStepNPY::getDirection()
{
    return glm::vec3(m_dirw);
}





void TorchStepNPY::setWeight(const char* s)
{
    m_dirw.w = boost::lexical_cast<float>(s) ;
}


// m_polw

void TorchStepNPY::setPolarization(const glm::vec4& pol)
{
    m_polw.x = pol.x ; 
    m_polw.y = pol.y ; 
    m_polw.z = pol.z ; 
}
void TorchStepNPY::setWavelength(const char* s)
{
    m_polw.w = boost::lexical_cast<float>(s) ;
}
float TorchStepNPY::getWavelength()
{
    return m_polw.w ; 
}
glm::vec3 TorchStepNPY::getPolarization()
{
    return glm::vec3(m_polw);
}





// m_zeaz

void TorchStepNPY::setZenithAzimuth(const char* s)
{
    std::string ss(s);
    m_zeaz = gvec4(ss) ;
}
glm::vec4 TorchStepNPY::getZenithAzimuth()
{
    return m_zeaz ; 
}



/// m_beam

void TorchStepNPY::setRadius(const char* s)
{
    setRadius(boost::lexical_cast<float>(s)) ;
}
void TorchStepNPY::setRadius(float radius)
{
    m_beam.x = radius ;
}
float TorchStepNPY::getRadius()
{
    return m_beam.x ; 
}



void TorchStepNPY::setDistance(const char* s)
{
    setDistance(boost::lexical_cast<float>(s)) ;
}
void TorchStepNPY::setDistance(float distance)
{
    m_beam.y = distance ;
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


    uif_t uif ; 
    uif.u = mode ; 
    m_beam.z = uif.f ;
}



void TorchStepNPY::dump(const char* msg)
{
    LOG(info) << msg  
              << " config " << m_config 
              << " material " << m_material
              ; 

    print(m_frame,              "m_frame ");
    print(m_source_local,       "m_source_local       ", m_src, "m_src");
    print(m_target_local,       "m_target_local       ", m_tgt, "m_tgt");
    print(m_polarization_local, "m_polarization_local ", m_pol, "m_pol");

    print(m_dir, "m_dir [normalize(m_tgt - m_src)] ");

    print(m_ctrl, "m_ctrl : id/pid/MaterialLine/NumPhotons" );
    print(m_post, "m_post : position, time " ); 
    print(m_dirw, "m_dirw : direction, weight" ); 
    print(m_polw, "m_polw : polarization, wavelength" ); 
    print(m_zeaz, "m_zeaz: zenith, azimuth " ); 
    print(m_beam, "m_beam: radius,... " ); 
}



void TorchStepNPY::Summary(const char* msg)
{
    glm::vec3 pos = getPosition() ;
    glm::vec3 dir = getDirection() ;
    glm::vec3 pol = getPolarization() ;

    LOG(info) << msg  
              << " typeName " << getTypeName() 
              << " modeString " << getModeString() 
              << " incidentSphere " << isIncidentSphere()
              << " sPolarized " << isSPolarized()
              << " pPolarized " << isPPolarized()
              << " position " << gformat(pos)
              << " direction " << gformat(dir)
              << " polarization " << gformat(pol)
              << " radius " << getRadius()
              << " wavelength " << getWavelength()
              << " time " << getTime()
              ; 

}


