#include "OrbitalView.hh"
#include "Animator.hh"
#include "NLog.hpp"

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


glm::vec4 OrbitalView::getEye(const glm::mat4& m2w) 
{ 
    return m_basis->getEye(m2w);
} 

glm::vec4 OrbitalView::getLook(const glm::mat4& m2w) 
{ 
    return m_basis->getLook(m2w);
} 

glm::vec4 OrbitalView::getUp(const glm::mat4& m2w) 
{ 
    return m_basis->getUp(m2w);
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




