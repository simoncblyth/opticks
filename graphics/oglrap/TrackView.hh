#pragma once

template <typename T> class NPY ;
#include "View.hh"

class Animator ; 

// operates from Composition via base class method View::getTransforms 
// which invokes the overriden getEye, getLook, getUp
// updating w2c c2w gaze
//
//
// Provides a head on animated view of the genstep, 
// does linear interpolation between start and end positions of the genstep.
// Look point if fixed at that advancing position with eye offset from
// there by time_ahead ns.  Initially the gaze if a head on view. Can manually 
// offset from there via rotations and translations.
//
//  See npy-/genstep.py for creation of the eg 1_track.npy file
//  that defines the path.
//  
//

class TrackView :  public View {
    public:
        static const char* PREFIX ; 
        virtual const char* getPrefix();
    public:
        TrackView(NPY<float>* track, unsigned int period=100, bool verbose=false);
        void Summary(const char* msg="TrackView::Summary");
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
        void gui();
    public:
        void setFraction(float fraction);
        std::string description(const char* msg="OV");
    private:
        void init();
        glm::vec4 getTrackPoint();
        glm::vec4 getTrackPoint(float fraction);
    private:
        unsigned int m_count ; 
        unsigned int m_period ; 
        float        m_fraction ; 
        Animator*    m_animator ;
        bool         m_verbose ; 

    private:
        NPY<float>*  m_track ;
        glm::vec4    m_origin ; 
        glm::vec4    m_direction ; 
        glm::vec4    m_range ; 
    private:
        glm::vec4    m_start ; 
        glm::vec4    m_end ; 
        float        m_time_ahead ; // ns ahead of the genstep 

};

inline TrackView::TrackView(NPY<float>* track, unsigned int period, bool verbose) 
     : 
     View(TRACK),
     m_count(0),
     m_period(period),
     m_fraction(0.f),
     m_animator(NULL),
     m_verbose(verbose),
     m_track(track),
     m_time_ahead(10.f)
{
    init();
}


inline void TrackView::setFraction(float fraction)
{
    m_fraction = fraction ; 
}


