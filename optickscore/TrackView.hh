/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#pragma once

/**
TrackView
============

Operates from Composition via base class method View::getTransforms 
which invokes the overriden getEye, getLook, getUp
updating w2c c2w gaze

Provides a head on animated view of the genstep, 
does linear interpolation between start and end positions of the genstep.
Look point if fixed at that advancing position with eye offset from
there by time_ahead ns.  Initially the gaze if a head on view. Can manually 
offset from there via rotations and translations.

See ana/genstep.py for creation of the eg 1_track.npy file
that defines the path.
  
**/

template <typename T> class NPY ;
class Animator ; 

#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"
#include "View.hh"

class OKCORE_API TrackView :  public View {
    public:
        static const char* PREFIX ; 
        virtual const char* getPrefix();
    public:
        TrackView(NPY<float>* track, unsigned int period=100, bool verbose=false);
        void Summary(const char* msg="TrackView::Summary");
        void setAnimator(Animator* animator);
        Animator* getAnimator();
    private:
        void initAnimator();
    public:
        // View overrides 
        glm::vec4 getEye(const glm::mat4& m2w);
        glm::vec4 getLook(const glm::mat4& m2w);
        glm::vec4 getUp(const glm::mat4& m2w);
        glm::vec4 getGaze(const glm::mat4& m2w, bool debug=false);
    public:
        float* getTEyeOffsetPtr();
        float* getTLookOffsetPtr();
        float* getTMinOffsetPtr();
        float* getTMaxOffsetPtr();
        float* getFractionScalePtr();
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
        float        m_teye_offset ; // ns ahead of the genstep 
        float        m_tlook_offset ; // ns ahead of the genstep 
        float        m_tmin_offset ; 
        float        m_tmax_offset ; 
        float        m_fraction_scale ; 
        bool         m_external ; 

};

#include "OKCORE_TAIL.hh"


