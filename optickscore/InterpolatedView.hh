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

#include <vector>
class Animator ; 
class SCtrl ; 

/**
InterpolatedView 
=================

Created by::

     Bookmarks::makeInterpolatedView
     FlightPath::makeInterpolatedView

operates from Composition via base class method View::getTransforms 
which invokes the overriden getEye, getLook, getUp
updating w2c c2w gaze

    void View::getTransforms(const glm::mat4& m2w, glm::mat4& world2camera, glm::mat4& camera2world, glm::vec4& gaze )


The view overrides all use interpolated fractional mixes between current and next view,
with the fraction being animated between  0 to 1 
before switching to the next pair of views.

**/

template<typename T> class NPY ; 

#include "View.hh"
#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

#include "plog/Severity.h"

class OKCORE_API InterpolatedView :  public View {
    public:
        static const char* PREFIX ; 
        static const plog::Severity LEVEL ; 
        virtual const char* getPrefix();
        static InterpolatedView* MakeFromArray(NPY<float>* elu, unsigned period, float scale, SCtrl* ctrl );
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
        void reset();
        void tick();
        void dump();
        bool isActive();
        bool hasChanged();
        void nextMode(unsigned int modifiers);
        void commandMode(const char* cmd);
    public:
        unsigned int getNumViews();
        void setFraction(float fraction);
        std::string description(const char* msg="IV");
        void setCtrl(SCtrl* ctrl); 
    private:
        void init();
        View* getView(unsigned int index);
        View* getCurrentView();
        View* getNextView();
        void nextPair();
        void setPair(unsigned int i, unsigned int j);
    private:
        unsigned  m_i ; 
        unsigned  m_j ; 
        unsigned  m_count ; 
        unsigned  m_period ; 
        float     m_fraction ; 
        Animator* m_animator ;
        std::vector<View*>  m_views ; 
        bool         m_verbose ; 
        SCtrl*       m_ctrl ; 

        unsigned  m_local_count ;
        const    glm::mat4 m_identity ; 

};

#include "OKCORE_TAIL.hh"

