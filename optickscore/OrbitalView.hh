#pragma once

#include <vector>
#include "View.hh"
class Animator ; 

// created by Composition::makeOrbitalView
//
// operates from Composition via base class method View::getTransforms 
// which invokes the overriden getEye, getLook, getUp
// updating w2c c2w gaze
//
// void View::getTransforms(const glm::mat4& m2w, glm::mat4& world2camera, glm::mat4& camera2world, glm::vec4& gaze )
//

#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

class OKCORE_API OrbitalView :  public View {
    public:
        static const char* PREFIX ; 
        virtual const char* getPrefix();
    public:
        OrbitalView(View* basis, unsigned int period=100, bool verbose=false);
        void Summary(const char* msg="OrbitalView::Summary");
        Animator* getAnimator();
    public:
        // View overrides 
        glm::vec4 getEye(const glm::mat4& m2w);
        glm::vec4 getLook(const glm::mat4& m2w);
        glm::vec4 getUp(const glm::mat4& m2w);
        glm::vec4 getGaze(const glm::mat4& m2w, bool debug=false);
    public:
        void tick();
        void nextMode(unsigned int modifiers);
        bool isActive();
        bool hasChanged();
    public:
        void setFraction(float fraction);
        std::string description(const char* msg="OV");
    private:
        void init();
        void update();
    private:
        View*        m_basis ;
        unsigned int m_count ; 
        unsigned int m_period ; 
        float        m_fraction ; 
        Animator*    m_animator ;
        bool         m_verbose ; 

    private:
        glm::vec4 m_orb_eye ; 
        glm::vec4 m_orb_look ; 
        glm::vec4 m_orb_up ; 


};

#include "OKCORE_TAIL.hh"

