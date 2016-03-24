#include "OrbitalView.hh"
#include "Animator.hh"
#include "NLog.hpp"
#include "GLMFormat.hpp"

#include <boost/lexical_cast.hpp>
#include <sstream>

#ifdef GUI_
#include <imgui.h>
#endif


const char* OrbitalView::PREFIX = "orbitalview" ;
const char* OrbitalView::getPrefix()
{
    return PREFIX ; 
}

void OrbitalView::init()
{
    m_animator = new Animator(&m_fraction, m_period, 0.f, 1.f ); 
    //m_animator->setModeRestrict(Animator::NORM);  // only OFF,SLOW,NORM,FAST, 
    if(m_verbose) m_animator->Summary("OrbitalView::init");
    m_animator->setMode(Animator::SLOW);
}

bool OrbitalView::hasChanged()
{
    return m_count > 0 && m_animator->isActive() ;  
}

void OrbitalView::nextMode(unsigned int modifiers)
{
    m_animator->nextMode(modifiers);
}

bool OrbitalView::isActive()
{
    return m_animator->isActive();
}


void OrbitalView::tick()
{
    m_count++ ; 

    bool bump(false);

    m_animator->step(bump);

    //LOG(info) << description("IV::tick") << " : " << m_animator->description() ;

    if(bump)
    {
        LOG(info) << description("OrbitalView::tick BUMP ") ; 
    }

    update();
}


std::string OrbitalView::description(const char* msg)
{
    std::stringstream ss ; 
    ss << msg 
       << " fr " << m_fraction 
       << " cn " << m_count
       ;
    return ss.str();
}


void OrbitalView::update()
{
    glm::vec4 base_m = m_basis->getEye() ;
    glm::vec3 tmp(base_m);
    tmp.z = 0.f ;  // hmm should dot product with up direction ?
    float r = glm::length(tmp) ; 

    float phi = m_fraction*float(M_PI)*2.0f  ;
    float sinphi = sin(phi);
    float cosphi = cos(phi);

    glm::vec4 hub = glm::vec4( 0.f , 0.f, base_m.z, 1.f );
    glm::vec4 gaze = glm::vec4( -r*sinphi, r*cosphi, 0.f, 0.f ) ;

    m_orb_eye  = glm::vec4( r*cosphi, r*sinphi, base_m.z, 1.f ) ;
    m_orb_look = m_orb_eye + gaze ; 
    m_orb_up = hub - m_orb_eye ; 

    LOG(debug) << "OrbitalView::update"
              << " base_m " << gformat(base_m)
              << " m_orb_eye " << gformat(m_orb_eye)
              << " m_orb_look " << gformat(m_orb_look)
              << " m_orb_up " << gformat(m_orb_up)
              << " r " << r 
              << " phi " << phi 
              << " fraction " << m_fraction
              ;
}


glm::vec4 OrbitalView::getEye(const glm::mat4& m2w) 
{ 
    if(m_count == 0) update();
    glm::vec4 eye_w = m2w * m_orb_eye ; 
    return eye_w ;  
} 

glm::vec4 OrbitalView::getLook(const glm::mat4& m2w) 
{ 
    if(m_count == 0) update();
    glm::vec4 look_w = m2w * m_orb_look ; 
    return look_w ;  
} 

glm::vec4 OrbitalView::getUp(const glm::mat4& m2w) 
{ 
    if(m_count == 0) update();
    glm::vec4 up_w = m2w * m_orb_up ; 
    return up_w ;  
} 

glm::vec4 OrbitalView::getGaze(const glm::mat4& m2w, bool debug)
{
    glm::vec4 eye = getEye(m2w);
    glm::vec4 look = getLook(m2w);
    glm::vec4 gaze = look - eye ; 
    return gaze ;                // w=0. OK as direction
}



void OrbitalView::Summary(const char* msg)
{
    LOG(info) << msg 
              ;

    m_basis->Summary(msg); 

}


void OrbitalView::gui()
{
#ifdef GUI_
    if(m_animator)
    {
         m_animator->gui("OrbitalView ", "%0.3f", 2.0f);
         ImGui::Text(" fraction %10.3f ", m_fraction  );
    }
#endif    
}




