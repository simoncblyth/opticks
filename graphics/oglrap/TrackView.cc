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

void TrackView::init()
{
    m_animator = new Animator(&m_fraction, m_period, 0.f, 1.f ); 
    //m_animator->setModeRestrict(Animator::NORM);  // only OFF,SLOW,NORM,FAST, 
    if(m_verbose) m_animator->Summary("TrackView::init");
    m_animator->setMode(Animator::SLOW);

    assert(m_track);

    m_origin    = m_track->getQuad(0,0) ;
    m_direction = m_track->getQuad(0,1) ;
    m_range     = m_track->getQuad(0,2) ;

    float tmin = m_range.x ; 
    float tmax = m_range.y ; 

    m_start      = m_origin + m_direction*tmin ; 
    m_end        = m_origin + m_direction*tmax ; 

    LOG(info) << "TrackView::init"
              << " track shape " << m_track->getShapeString()
              << " origin " << gformat(m_origin)
              << " direction " << gformat(m_direction)
              << " range " << gformat(m_range)
              << " start " << gformat(m_start)
              << " end " << gformat(m_end)
              ;

}

bool TrackView::hasChanged()
{
    return m_count > 0 && m_animator->isActive() ;  
}

void TrackView::nextMode(unsigned int modifiers)
{
    m_animator->nextMode(modifiers);
}

bool TrackView::isActive()
{
    return m_animator->isActive();
}


void TrackView::tick()
{
    m_count++ ; 

    bool bump(false);

    m_animator->step(bump);

    //LOG(info) << description("IV::tick") << " : " << m_animator->description() ;

    if(bump)
    {
        LOG(info) << description("TrackView::tick BUMP ") ; 
    }
}


std::string TrackView::description(const char* msg)
{
    std::stringstream ss ; 
    ss << msg 
       << " fr " << m_fraction 
       << " cn " << m_count
       ;
    return ss.str();
}

glm::vec4 TrackView::getTrackPoint()
{
   return getTrackPoint(m_fraction);
} 

glm::vec4 TrackView::getTrackPoint(float fraction) 
{ 
    glm::vec3 a = glm::vec3(m_start) ; 
    glm::vec3 b = glm::vec3(m_end) ; 
    glm::vec3 m = glm::mix(a,b,fraction); 
    return glm::vec4(m.x, m.y, m.z, 1.0f ) ; 
}

glm::vec4 TrackView::getEye(const glm::mat4& m2w) 
{ 
    glm::vec4 trkp = getTrackPoint(); 
    return trkp + m_time_ahead*m_direction  ; 
}

glm::vec4 TrackView::getLook(const glm::mat4& m2w) 
{ 
    glm::vec4 trkp = getTrackPoint(); 
    return trkp ; 
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
    if(m_animator)
    {
         m_animator->gui("TrackView ", "%0.3f", 2.0f);
         ImGui::Text(" fraction %10.3f ", m_fraction  );
    }
    ImGui::SliderFloat("time ahead (ns)", &m_time_ahead, -20.0f, 20.0f);

#endif    
}


