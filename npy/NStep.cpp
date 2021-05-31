#include <sstream>
#include "BStr.hh"
#include "GLMFormat.hpp"
#include "NPY.hpp"
#include "NStep.hpp"

#include "PLOG.hh"


const plog::Severity NStep::LEVEL = PLOG::EnvLevel("NStep", "DEBUG"); 

NStep::NStep()
    :
    m_array(NPY<float>::make(1, 6, 4)),
    m_filled(false),
    m_ctrl(0,0,0,0),
    m_post(0,0,0,0),
    m_dirw(0,0,0,0),
    m_polw(0,0,0,0),
    m_zeaz(0,0,0,0),
    m_beam(0,0,0,0)
{
    m_array->zero(); 
}

void NStep::fillArray() 
{
    m_filled = true ; 
    unsigned i = 0 ; 
    m_array->setQuadI(m_ctrl, i, 0);
    m_array->setQuad( m_post, i, 1);
    m_array->setQuad( m_dirw, i, 2);
    m_array->setQuad( m_polw, i, 3);
    m_array->setQuad( m_zeaz, i, 4);
    m_array->setQuad( m_beam, i, 5);
}

NPY<float>* NStep::getArray() const 
{
    return m_filled ? m_array : nullptr ; 
}


// m_ctrl

void NStep::setGenstepType(unsigned gentype)
{
    m_ctrl.x = gentype ;  // eg OpticksGenstep_TORCH
}
unsigned NStep::getGenstepType() const 
{
    return m_ctrl.x ; 
}

/**
NStep::setOriginTrackID
------------------------------

m_ctrl.y seems to be unused by torch gensteps, 
but it is used for OriginTrackID for "real" gensteps
hence for debugging with for example with g4ok/tests/G4OKTest.cc
it is handy to be able to set this.

**/

void NStep::setOriginTrackID(unsigned trackID)
{
    m_ctrl.y = trackID ;  // eg OpticksGenstep_TORCH
}
unsigned NStep::getOriginTrackID() const 
{
    return m_ctrl.y ; 
}

void NStep::setMaterialLine(unsigned int ml)
{
    m_ctrl.z = ml ; 
}
unsigned NStep::getMaterialLine() const
{
    return m_ctrl.z ;
}


void NStep::setNumPhotons(const char* s)
{
    setNumPhotons(BStr::LexicalCast<unsigned int>(s)) ; 
}
void NStep::setNumPhotons(unsigned int num_photons)
{
    m_ctrl.w = num_photons ; 
}
unsigned int NStep::getNumPhotons() const 
{
    return m_ctrl.w ; 
}






// m_post

void NStep::setPosition(const glm::vec4& pos)
{
    m_post.x = pos.x ; 
    m_post.y = pos.y ; 
    m_post.z = pos.z ; 
}

void NStep::setTime(const char* s)
{
    m_post.w = BStr::LexicalCast<float>(s) ;
}
float NStep::getTime() const
{
    return m_post.w ; 
}

glm::vec3 NStep::getPosition() const 
{
    return glm::vec3(m_post);
}




// m_dirw

void NStep::setDirection(const char* s)
{
    std::string ss(s);
    glm::vec3 dir = gvec3(ss) ;
    setDirection(dir);
}

void NStep::setDirection(const glm::vec3& dir)
{
    m_dirw.x = dir.x ; 
    m_dirw.y = dir.y ; 
    m_dirw.z = dir.z ; 
}

glm::vec3 NStep::getDirection() const 
{
    return glm::vec3(m_dirw);
}

void NStep::setWeight(const char* s)
{
    m_dirw.w = BStr::LexicalCast<float>(s) ;
}


// m_polw

void NStep::setPolarization(const glm::vec4& pol)
{
    glm::vec4 npol = glm::normalize(pol);

    m_polw.x = npol.x ; 
    m_polw.y = npol.y ; 
    m_polw.z = npol.z ; 

    LOG(LEVEL)
        << " pol " << gformat(pol)
        << " npol " << gformat(npol)
        << " m_polw " << gformat(m_polw)
        ;

    //assert(0); 
}
void NStep::setWavelength(const char* s)
{
    m_polw.w = BStr::LexicalCast<float>(s) ;
}
float NStep::getWavelength() const 
{
    return m_polw.w ; 
}
glm::vec3 NStep::getPolarization() const 
{
    return glm::vec3(m_polw);
}





// m_zeaz

void NStep::setZenithAzimuth(const char* s)
{
    std::string ss(s);
    m_zeaz = gvec4(ss) ;
}
glm::vec4 NStep::getZenithAzimuth() const
{
    return m_zeaz ; 
}



/// m_beam

void NStep::setRadius(const char* s)
{
    setRadius(BStr::LexicalCast<float>(s)) ;
}
void NStep::setRadius(float radius)
{
    m_beam.x = radius ;
}
float NStep::getRadius() const 
{
    return m_beam.x ; 
}

void NStep::setDistance(const char* s)
{
    setDistance(BStr::LexicalCast<float>(s)) ;
}
void NStep::setDistance(float distance)
{
    m_beam.y = distance ;
}

void NStep::setBaseMode(unsigned umode)
{
    uif_t uif ; 
    uif.u = umode ; 
    m_beam.z = uif.f ;
}
unsigned NStep::getBaseMode() const
{
    uif_t uif ;
    uif.f = m_beam.z ; 
    return uif.u ; 
}


void NStep::setBaseType(unsigned utype)
{
    uif_t uif ; 
    uif.u = utype ; 
    m_beam.w = uif.f ;
}
unsigned NStep::getBaseType() const 
{
    uif_t uif ;
    uif.f = m_beam.w ; 
    return uif.u ; 
}







std::string NStep::desc(const char* msg) const 
{
    std::stringstream ss ; 

    int wid = 10 ; 
    int prec = 3 ; 

    ss << msg << std::endl ; 
    ss << GLMFormat::Format(m_ctrl, wid, 0)    << "m_ctrl : id/pid/MaterialLine/NumPhotons" << std::endl; 
    ss << GLMFormat::Format(m_post, wid, prec) << "m_post : position, time "  << std::endl ; 
    ss << GLMFormat::Format(m_dirw, wid, prec) << "m_dirw : direction, weight" << std::endl ; 
    ss << GLMFormat::Format(m_polw, wid, prec) << "m_polw : polarization, wavelength" << std::endl ; 
    ss << GLMFormat::Format(m_zeaz, wid, prec) << "m_zeaz: zenith, azimuth " << std::endl ; 
    ss << GLMFormat::Format(m_beam, wid, prec) << "m_beam: radius,... " << std::endl ; 

    std::string s = ss.str(); 
    return s ; 
}



