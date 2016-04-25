#pragma once

#include <vector>
#include "View.hh"
class Animator ; 

// created by Bookmarks::makeInterpolatedView
//
// operates from Composition via base class method View::getTransforms 
// which invokes the overriden getEye, getLook, getUp
// updating w2c c2w gaze
//
// void View::getTransforms(const glm::mat4& m2w, glm::mat4& world2camera, glm::mat4& camera2world, glm::vec4& gaze )
//
//

class InterpolatedView :  public View {
    public:
        static const char* PREFIX ; 
        virtual const char* getPrefix();
    public:
        InterpolatedView(unsigned int period=100, bool verbose=false);
        Animator* getAnimator();
        void addView(View* view);
        void Summary(const char* msg="View::Summary");
    public:
        // View overrides 
        glm::vec4 getEye(const glm::mat4& m2w);
        glm::vec4 getLook(const glm::mat4& m2w);
        glm::vec4 getUp(const glm::mat4& m2w);
        glm::vec4 getGaze(const glm::mat4& m2w, bool debug=false);
    public:
        void tick();
        bool isActive();
        bool hasChanged();
        void nextMode(unsigned int modifiers);
    public:
        unsigned int getNumViews();
        void setFraction(float fraction);
        std::string description(const char* msg="IV");
    private:
        void init();
        View* getView(unsigned int index);
        View* getCurrentView();
        View* getNextView();
        void nextPair();
        void setPair(unsigned int i, unsigned int j);
    private:
        unsigned int m_i ; 
        unsigned int m_j ; 
        unsigned int m_count ; 
        unsigned int m_period ; 
        float        m_fraction ; 
        Animator*    m_animator ;
        std::vector<View*>  m_views ; 
        bool         m_verbose ; 

};

inline InterpolatedView::InterpolatedView(unsigned int period, bool verbose) 
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



inline Animator* InterpolatedView::getAnimator()
{
    return m_animator ; 
}

inline void InterpolatedView::addView(View* view)
{
    m_views.push_back(view);
}
inline unsigned int InterpolatedView::getNumViews()
{
   return m_views.size();
}
inline View* InterpolatedView::getView(unsigned int index)
{
    return index < getNumViews() ? m_views[index] : NULL ;
}
inline View* InterpolatedView::getCurrentView()
{
    return getView(m_i);
}
inline View* InterpolatedView::getNextView()
{
    return getView(m_j);
}

inline void InterpolatedView::setFraction(float fraction)
{
    m_fraction = fraction ; 
}

inline void InterpolatedView::setPair(unsigned int i, unsigned int j)
{
    m_i = i ;
    m_j = j ; 
}

inline void InterpolatedView::nextPair()
{
    unsigned int n = getNumViews();
    unsigned int i = (m_i + 1) % n ;   
    unsigned int j = (m_j + 1) % n ;
    setPair(i,j);
}



