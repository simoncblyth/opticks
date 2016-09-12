
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>

// npy-
#include "GenstepNPY.hpp"
#include "NPY.hpp"
#include "GLMPrint.hpp"
#include "GLMFormat.hpp"
#include "uif.h"

#include "PLOG.hh"


GenstepNPY* GenstepNPY::Fabricate(unsigned genstep_type, unsigned num_step, unsigned num_photons_per_step)
{
    GenstepNPY* fab = new GenstepNPY(genstep_type, num_step) ;      
    for(unsigned i=0 ; i < num_step ; i++)
    {   
        fab->setMaterialLine(i*10);   
        fab->setNumPhotons(num_photons_per_step); 
        fab->addStep();
    }   
    return fab ; 
}


GenstepNPY::GenstepNPY(unsigned genstep_type, unsigned num_step) 
       :  
       m_genstep_type(genstep_type),
       m_npy(NPY<float>::make(num_step, 6, 4)),
       m_num_step(num_step),
       m_step_index(0)
{
    m_npy->zero();
}

void GenstepNPY::update()
{
    // placeholder overridden in subclasses like TorchStepNPY 
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






void GenstepNPY::dump(const char* msg)
{
    LOG(info) << msg  
              ; 

    print(m_ctrl, "m_ctrl : id/pid/MaterialLine/NumPhotons" );
    print(m_post, "m_post : position, time " ); 
    print(m_dirw, "m_dirw : direction, weight" ); 
    print(m_polw, "m_polw : polarization, wavelength" ); 
    print(m_zeaz, "m_zeaz: zenith, azimuth " ); 
    print(m_beam, "m_beam: radius,... " ); 
}







