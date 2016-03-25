#include "TrackView.hh"
#include "Animator.hh"

// npy-
#include "NLog.hpp"
#include "NPY.hpp"
#include "GLMFormat.hpp"

#include <boost/lexical_cast.hpp>
#include <sstream>

#ifdef GUI_
#include <imgui.h>
#endif


const char* TrackView::PREFIX = "trackview" ;
const char* TrackView::getPrefix()
{
    return PREFIX ; 
}


void TrackView::setAnimator(Animator* animator)
{
    m_animator = animator ; 
    m_external = true ; 

}
Animator* TrackView::getAnimator()
{   
    if(!m_animator) initAnimator();
    return m_animator ; 
}
void TrackView::initAnimator()
{
    m_animator = new Animator(&m_fraction, m_period, 0.f, 1.f ); 
    //m_animator->setModeRestrict(Animator::NORM);  // only OFF,SLOW,NORM,FAST, 
    if(m_verbose) m_animator->Summary("TrackView::initAnimator");
    m_animator->setMode(Animator::SLOW);
    m_external = false ; 
}


void TrackView::init()
{
    assert(m_track);

    m_origin    = m_track->getQuad(0,0) ;
    m_direction = m_track->getQuad(0,1) ;
    m_range     = m_track->getQuad(0,2) ;

    LOG(info) << "TrackView::init"
              << " track shape " << m_track->getShapeString()
              << " origin " << gformat(m_origin)
              << " direction " << gformat(m_direction)
              << " range " << gformat(m_range)
              ;
}

bool TrackView::hasChanged()
{
    Animator* animator = getAnimator();
    return m_count > 0 && animator->isActive() ;  
}

void TrackView::nextMode(unsigned int modifiers)
{
    Animator* animator = getAnimator();
    animator->nextMode(modifiers);
}

bool TrackView::isActive()
{
    Animator* animator = getAnimator();
    return animator->isActive();
}


void TrackView::tick()
{
    if(m_external) return ; 

    m_count++ ; 

    bool bump(false);

    Animator* animator = getAnimator();
    animator->step(bump);

    if(bump)
    {
        LOG(info) << description("TrackView::tick BUMP ") ; 
    }
}


std::string TrackView::description(const char* msg)
{
    std::stringstream ss ; 
    ss << msg 
       << " cn " << m_count
       ;
    return ss.str();
}

glm::vec4 TrackView::getTrackPoint()
{
   Animator* animator = getAnimator(); 
   float fraction = animator->getFractionFromTarget();
   glm::vec4 tkp = getTrackPoint(fraction);

   LOG(debug) << "TrackView::getTrackPoint"
             << " fraction " << fraction 
             << " tkp " << gformat(tkp)
             ; 
   return tkp ; 
} 

glm::vec4 TrackView::getTrackPoint(float fraction) 
{ 
    float tmin = m_range.x ; 
    float tmax = m_range.y ; 

    glm::vec4 start = m_origin + m_direction*(tmin + m_tmin_offset) ; 
    glm::vec4 end   = m_origin + m_direction*(tmax + m_tmax_offset) ; 
    glm::vec3 a = glm::vec3(start) ; 
    glm::vec3 b = glm::vec3(end) ; 
    glm::vec3 m = glm::mix(a,b,fraction*m_fraction_scale); 

    return glm::vec4(m.x, m.y, m.z, 1.0f ) ; 
}

glm::vec4 TrackView::getEye(const glm::mat4& m2w) 
{ 
    glm::vec4 trkp = getTrackPoint(); 
    return trkp + m_teye_offset*m_direction  ; 
}

glm::vec4 TrackView::getLook(const glm::mat4& m2w) 
{ 
    glm::vec4 trkp = getTrackPoint(); 
    return trkp + m_tlook_offset*m_direction ; 
} 

glm::vec4 TrackView::getUp(const glm::mat4& m2w) 
{ 
    return glm::vec4(0,0,1000,0) ;  
} 

glm::vec4 TrackView::getGaze(const glm::mat4& m2w, bool debug)
{
    glm::vec4 eye = getEye(m2w);
    glm::vec4 look = getLook(m2w);
    glm::vec4 gaze = look - eye ; 
    return gaze ;                // w=0. OK as direction
}


void TrackView::Summary(const char* msg)
{
    LOG(info) << msg 
              ;

    m_track->dump(msg); 
}


void TrackView::gui()
{
#ifdef GUI_
    Animator* animator = getAnimator();
    if(animator)
    {
         animator->gui("TrackView ", "%0.3f", 2.0f);
         //ImGui::Text(" fraction %10.3f ", m_fraction  );
    }
    ImGui::SliderFloat("tmin offset (ns)", &m_tmin_offset, -20.0f, 20.0f);
    ImGui::SliderFloat("tmax offset (ns)", &m_tmax_offset, -20.0f, 20.0f);
    ImGui::SliderFloat("teye offset (ns)", &m_teye_offset, -20.0f, 50.0f);
    ImGui::SliderFloat("tlook offset (ns)", &m_tlook_offset, -20.0f, 50.0f);
    ImGui::SliderFloat("fraction scale",    &m_fraction_scale, 1.0f, 2.0f);

#endif    
}


