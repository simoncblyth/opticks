#pragma once

#include <vector>
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
#include "View.hh"
#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

class OKCORE_API InterpolatedView :  public View {
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

#include "OKCORE_TAIL.hh"

