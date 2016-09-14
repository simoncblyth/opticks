
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>

// npy-
#include "GenstepNPY.hpp"
#include "NPY.hpp"
#include "GLMPrint.hpp"
#include "GLMFormat.hpp"
#include "uif.h"

#include "PLOG.hh"




GenstepNPY::GenstepNPY(unsigned genstep_type, unsigned num_step, const char* config) 
       :  
       m_genstep_type(genstep_type),
       m_num_step(num_step),
       m_config(config ? strdup(config) : NULL),
       m_material(NULL),
       m_npy(NPY<float>::make(num_step, 6, 4)),
       m_step_index(0),
       m_frame_targetted(false)
{
    m_npy->zero();
}

void GenstepNPY::addActionControl(unsigned long long  action_control)
{
    m_npy->addActionControl(action_control);
}

const char* GenstepNPY::getMaterial()
{
    return m_material ; 
}
const char* GenstepNPY::getConfig()
{
    return m_config ; 
}

void GenstepNPY::setMaterial(const char* s)
{
    m_material = strdup(s);
}

unsigned GenstepNPY::getNumStep()
{
   return m_num_step ;  
}

void GenstepNPY::addStep(bool verbose)
{
    assert(m_npy && m_npy->hasData());

    unsigned int i = m_step_index ; 

    setGenstepType( m_genstep_type ) ;    
    update(); 

    if(verbose) dump("GenstepNPY::addStep");

    m_npy->setQuadI(m_ctrl, i, 0 );
    m_npy->setQuad( m_post, i, 1);
    m_npy->setQuad( m_dirw, i, 2);
    m_npy->setQuad( m_polw, i, 3);
    m_npy->setQuad( m_zeaz, i, 4);
    m_npy->setQuad( m_beam, i, 5);

    m_step_index++ ; 
}

NPY<float>* GenstepNPY::getNPY()
{
    assert( m_step_index == m_num_step && "GenstepNPY is incomplete, must addStep according to declared num_step");
    return m_npy ; 
}







// m_ctrl

void GenstepNPY::setGenstepType(unsigned genstep_type)
{
    m_ctrl.x = genstep_type ;  // eg TORCH
}
void GenstepNPY::setMaterialLine(unsigned int ml)
{
    m_ctrl.z = ml ; 
}


void GenstepNPY::setNumPhotons(const char* s)
{
    setNumPhotons(boost::lexical_cast<unsigned int>(s)) ; 
}
void GenstepNPY::setNumPhotons(unsigned int num_photons)
{
    m_ctrl.w = num_photons ; 
}
unsigned int GenstepNPY::getNumPhotons()
{
    return m_ctrl.w ; 
}



// m_post

void GenstepNPY::setPosition(const glm::vec4& pos)
{
    m_post.x = pos.x ; 
    m_post.y = pos.y ; 
    m_post.z = pos.z ; 
}

void GenstepNPY::setTime(const char* s)
{
    m_post.w = boost::lexical_cast<float>(s) ;
}
float GenstepNPY::getTime()
{
    return m_post.w ; 
}

glm::vec3 GenstepNPY::getPosition()
{
    return glm::vec3(m_post);
}




// m_dirw

void GenstepNPY::setDirection(const char* s)
{
    std::string ss(s);
    glm::vec3 dir = gvec3(ss) ;
    setDirection(dir);
}

void GenstepNPY::setDirection(const glm::vec3& dir)
{
    m_dirw.x = dir.x ; 
    m_dirw.y = dir.y ; 
    m_dirw.z = dir.z ; 
}

glm::vec3 GenstepNPY::getDirection()
{
    return glm::vec3(m_dirw);
}












void GenstepNPY::setWeight(const char* s)
{
    m_dirw.w = boost::lexical_cast<float>(s) ;
}


// m_polw

void GenstepNPY::setPolarization(const glm::vec4& pol)
{
    m_polw.x = pol.x ; 
    m_polw.y = pol.y ; 
    m_polw.z = pol.z ; 
}
void GenstepNPY::setWavelength(const char* s)
{
    m_polw.w = boost::lexical_cast<float>(s) ;
}
float GenstepNPY::getWavelength()
{
    return m_polw.w ; 
}
glm::vec3 GenstepNPY::getPolarization()
{
    return glm::vec3(m_polw);
}





// m_zeaz

void GenstepNPY::setZenithAzimuth(const char* s)
{
    std::string ss(s);
    m_zeaz = gvec4(ss) ;
}
glm::vec4 GenstepNPY::getZenithAzimuth()
{
    return m_zeaz ; 
}



/// m_beam

void GenstepNPY::setRadius(const char* s)
{
    setRadius(boost::lexical_cast<float>(s)) ;
}
void GenstepNPY::setRadius(float radius)
{
    m_beam.x = radius ;
}
float GenstepNPY::getRadius()
{
    return m_beam.x ; 
}



void GenstepNPY::setDistance(const char* s)
{
    setDistance(boost::lexical_cast<float>(s)) ;
}
void GenstepNPY::setDistance(float distance)
{
    m_beam.y = distance ;
}



unsigned GenstepNPY::getBaseMode()
{
    uif_t uif ;
    uif.f = m_beam.z ; 
    return uif.u ; 
}
void GenstepNPY::setBaseMode(unsigned umode)
{
    uif_t uif ; 
    uif.u = umode ; 
    m_beam.z = uif.f ;
}




void GenstepNPY::setBaseType(unsigned utype)
{
    uif_t uif ; 
    uif.u = utype ; 
    m_beam.w = uif.f ;
}

unsigned GenstepNPY::getBaseType()
{
    uif_t uif ;
    uif.f = m_beam.w ; 
    return uif.u ; 
}










void GenstepNPY::setFrameTransform(const char* s)
{
    std::string ss(s);
    bool flip = true ;  
    glm::mat4 transform = gmat4(ss, flip);
    setFrameTransform(transform);
    setFrameTargetted(true);
}


void GenstepNPY::setFrameTransform(glm::mat4& frame_transform)
{
    m_frame_transform = frame_transform ;
}
const glm::mat4& GenstepNPY::getFrameTransform()
{
    return m_frame_transform ;
}


void GenstepNPY::setFrameTargetted(bool targetted)
{
    m_frame_targetted = targetted ;
}
bool GenstepNPY::isFrameTargetted()
{
    return m_frame_targetted ;
} 

void GenstepNPY::setFrame(const char* s)
{
    std::string ss(s);
    m_frame = givec4(ss);
}
void GenstepNPY::setFrame(unsigned int vindex)
{
    m_frame.x = vindex ; 
    m_frame.y = 0 ; 
    m_frame.z = 0 ; 
    m_frame.w = 0 ; 
}
glm::ivec4& GenstepNPY::getFrame()
{
    return m_frame ; 
}





void GenstepNPY::dump(const char* msg)
{
    dumpBase(msg);
}

void GenstepNPY::dumpBase(const char* msg)
{
    LOG(info) << msg  
              << " config " << m_config 
              << " material " << m_material
              ; 

    print(m_ctrl, "m_ctrl : id/pid/MaterialLine/NumPhotons" );
    print(m_post, "m_post : position, time " ); 
    print(m_dirw, "m_dirw : direction, weight" ); 
    print(m_polw, "m_polw : polarization, wavelength" ); 
    print(m_zeaz, "m_zeaz: zenith, azimuth " ); 
    print(m_beam, "m_beam: radius,... " ); 

    print(m_frame, "m_frame ");
}







