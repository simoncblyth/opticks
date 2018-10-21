#include <boost/lexical_cast.hpp>
#include <sstream>

#include "PLOG.hh"

#include "NGLM.hpp"

#include "Animator.hh"
#include "InterpolatedView.hh"


const char* InterpolatedView::PREFIX = "interpolatedview" ;
const char* InterpolatedView::getPrefix()
{
    return PREFIX ; 
}


InterpolatedView::InterpolatedView(unsigned int period, bool verbose) 
     : 
     View(INTERPOLATED),
     m_i(0),
     m_j(1),
     m_count(0),
     m_period(period),
     m_fraction(0.f),
     m_animator(NULL),
     m_verbose(verbose)
{
    init();
}



Animator* InterpolatedView::getAnimator()
{
    return m_animator ; 
}

void InterpolatedView::addView(View* view)
{
    m_views.push_back(view);
}
unsigned int InterpolatedView::getNumViews()
{
   return m_views.size();
}
View* InterpolatedView::getView(unsigned int index)
{
    return index < getNumViews() ? m_views[index] : NULL ;
}
View* InterpolatedView::getCurrentView()
{
    return getView(m_i);
}
View* InterpolatedView::getNextView()
{
    return getView(m_j);
}

void InterpolatedView::setFraction(float fraction)
{
    m_fraction = fraction ; 
}

void InterpolatedView::setPair(unsigned int i, unsigned int j)
{
    m_i = i ;
    m_j = j ; 
}

void InterpolatedView::nextPair()
{
    unsigned int n = getNumViews();
    unsigned int i = (m_i + 1) % n ;   
    unsigned int j = (m_j + 1) % n ;
    setPair(i,j);
}


void InterpolatedView::init()
{
    m_animator = new Animator(&m_fraction, m_period, 0.f, 1.f ); 
    //m_animator->setModeRestrict(Animator::NORM);  // only OFF,SLOW,NORM,FAST, 
    if(m_verbose) m_animator->Summary("InterpolatedView::init");
    //m_animator->setMode(Animator::SLOW4);
    m_animator->setMode(Animator::SLOW2);
}

bool InterpolatedView::hasChanged()
{
    return m_count > 0 && m_animator->isActive() ;  
}

void InterpolatedView::nextMode(unsigned int modifiers)
{
    m_animator->nextMode(modifiers);
}

bool InterpolatedView::isActive()
{
    return m_animator->isActive();
}



void InterpolatedView::tick()
{
    m_count++ ; 

    bool bump(false);

    m_animator->step(bump);

    LOG(info) << description("IV::tick") << " : " << m_animator->description() ;

    if(bump)
    {
        nextPair();
        LOG(info) << description("InterpolatedView::tick BUMP ") ; 
    }
}


std::string InterpolatedView::description(const char* msg)
{
    std::stringstream ss ; 
    ss << msg << " [" << getNumViews() << "] (" <<  m_i << "," << m_j << ") " << m_fraction << " " << m_count  ;
    return ss.str();
}


glm::vec4 InterpolatedView::getEye(const glm::mat4& m2w) 
{ 
    View* curr = getCurrentView() ;
    View* next = getNextView() ;
    glm::vec3 a = glm::vec3(curr->getEye(m2w)) ; 
    glm::vec3 b = glm::vec3(next->getEye(m2w)) ; 
    glm::vec3 m = glm::mix(a,b,m_fraction); 

    return glm::vec4(m.x, m.y, m.z, 1.0f ) ; 
} 

glm::vec4 InterpolatedView::getLook(const glm::mat4& m2w) 
{ 
    View* curr = getCurrentView() ;
    View* next = getNextView() ;
    glm::vec3 a = glm::vec3(curr->getLook(m2w)) ; 
    glm::vec3 b = glm::vec3(next->getLook(m2w)) ; 
    glm::vec3 m = glm::mix(a,b,m_fraction); 
    return glm::vec4(m.x, m.y, m.z, 1.0f ) ; 
} 

glm::vec4 InterpolatedView::getUp(const glm::mat4& m2w) 
{ 
    View* curr = getCurrentView() ;
    View* next = getNextView() ;
    glm::vec3 a = glm::vec3(curr->getUp(m2w)) ; 
    glm::vec3 b = glm::vec3(next->getUp(m2w)) ; 
    glm::vec3 m = glm::mix(a,b,m_fraction); 
    return glm::vec4(m.x, m.y, m.z, 0.0f ) ; // w=0 as direction
} 

glm::vec4 InterpolatedView::getGaze(const glm::mat4& m2w, bool )
{
    glm::vec4 eye = getEye(m2w);
    glm::vec4 look = getLook(m2w);
    glm::vec4 gaze = look - eye ; 
    return gaze ;                // w=0. OK as direction
}



void InterpolatedView::Summary(const char* msg)
{
    unsigned int nv = getNumViews();
    LOG(info) << msg 
              << " NumViews " << nv ; 

    for(unsigned int i=0 ; i < nv ; i++)
    {
        View* v = getView(i);
        std::string vmsg = boost::lexical_cast<std::string>(i);
        v->Summary(vmsg.c_str());
    }
}


