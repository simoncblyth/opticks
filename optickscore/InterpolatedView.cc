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

#include <boost/lexical_cast.hpp>
#include <sstream>


#include "SCtrl.hh"
#include "BStr.hh"
#include "PLOG.hh"

#include "NPY.hpp"
#include "NGLM.hpp"

#include "Animator.hh"
#include "InterpolatedView.hh"


const plog::Severity InterpolatedView::LEVEL = PLOG::EnvLevel("InterpolatedView", "DEBUG" ); 

const char* InterpolatedView::PREFIX = "interpolatedview" ;
const char* InterpolatedView::getPrefix()
{
    return PREFIX ; 
}



InterpolatedView* InterpolatedView::MakeFromArray(NPY<float>* elu, unsigned period, SCtrl* ctrl )
{
    LOG(LEVEL) << "[" ; 
    assert( elu && elu->hasShape(-1,4,4) ); 

    if(elu->getNumItems() < 2)
    {
        LOG(fatal) << " requires at least 2 views " ;
        assert(0);  
    }

    InterpolatedView* iv = new InterpolatedView(period) ; 
    iv->setCtrl(ctrl); 

    for(unsigned i=0 ; i < elu->getNumItems() ; i++ )
    {
        View* v = View::FromArrayItem( elu, i ) ; 
        iv->addView(v);
    }

    iv->reset(); 

    LOG(LEVEL) << "]" ; 
    return iv ; 
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
    m_verbose(verbose),
    m_ctrl(NULL),
    m_local_count(-1),
    m_identity(1.f)
{
    init();
}

void InterpolatedView::init()
{
    LOG(LEVEL); 
    m_animator = new Animator(&m_fraction, m_period, 0.f, 1.f, "InterpolatedView" ); 
    //m_animator->setModeRestrict(Animator::NORM);  // only OFF,SLOW,NORM,FAST, 
    if(m_verbose) m_animator->Summary("InterpolatedView::init");
    //m_animator->setMode(Animator::SLOW4);
    m_animator->setMode(Animator::SLOW2);
}

void InterpolatedView::reset()
{
    LOG(LEVEL); 
    m_count = 0 ; 
    m_fraction = 0.f ;  // start from entirely currentView 0
    setPair(0,1);  
    m_animator->home(); 
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

void InterpolatedView::setPair(unsigned i, unsigned j)
{
    m_i = i ;
    m_j = j ; 
}

void InterpolatedView::setCtrl(SCtrl* ctrl)
{
    m_ctrl = ctrl ; 
}

void InterpolatedView::nextPair()
{
    //LOG(LEVEL); 
    unsigned int n = getNumViews();
    unsigned int i = (m_i + 1) % n ;   
    unsigned int j = (m_j + 1) % n ;
    setPair(i,j);
}


bool InterpolatedView::hasChanged()
{
    return m_count > 0 && m_animator->isActive() ;  
}

void InterpolatedView::nextMode(unsigned int modifiers)
{
    LOG(info) << description() ; 
    m_animator->nextMode(modifiers);
}
void InterpolatedView::commandMode(const char* cmd)
{
    m_animator->commandMode(cmd);
}


bool InterpolatedView::isActive()
{
    return m_animator->isActive();
}


void InterpolatedView::dump()
{
    glm::vec4 e = getEye(m_identity); 
    glm::vec4 l = getLook(m_identity); 
    glm::vec4 u = getUp(m_identity); 

    std::cout 
        << description("IV.dump") 
        << " e (" << e.x << "," << e.y << "," << e.z << "," << e.w << ") " 
        << " l (" << l.x << "," << l.y << "," << l.z << "," << l.w << ") " 
        << " u (" << u.x << "," << u.y << "," << u.z << "," << u.w << ") " 
        << std::endl 
        ; 

}



/**
InterpolatedView::tick
-----------------------------------

Commands are passed to the appropriate objects 
from the high controller, using the SCtrl mechanism.

**/

void InterpolatedView::tick()
{
    LOG(LEVEL) ; 
    //dump(); 

    m_count++ ; 
    m_local_count++ ; 

    bool bump(false);
    unsigned cmd_index(0) ; 
    unsigned cmd_offset(0) ; 

    m_animator->step(bump, cmd_index, cmd_offset);
    
    // spread out command dispatch for a view according to their slots
    // for better control of, factor of 8 better granularity.
 
    if(m_ctrl && cmd_offset == 1)
    { 
        View* curr = getCurrentView() ; 
        if(curr->hasCmds())  
        {
            const std::string& cmd = curr->getCmd(cmd_index) ; 
            if( cmd.compare("  ") != 0 )  
            {
                m_ctrl->command(cmd.c_str());  

                LOG(info) 
                    << " cmd " << cmd 
                    << " ("
                    << " cmd_index " << cmd_index
                    << " cmd_offset " << cmd_offset
                    << " )"
                    ;
            }
        }
    }

    //LOG(info) << description("IV::tick") << " : " << m_animator->description() ;
    //LOG(info) << description("IV") ; 

    if(bump)
    {
        //dump(); 
        nextPair();
        m_local_count = -1 ;  
    }
}


std::string InterpolatedView::description(const char* msg)
{
    std::stringstream ss ; 
    ss << msg << " [" << getNumViews() << "] (" <<  m_i << "," << m_j << ")"
       << " f:" << std::setw(10) << std::fixed << std::setprecision(4) << m_fraction 
       << " c:" << std::setw(6) << m_count 
       << " lc:" << std::setw(6) << int(m_local_count)
       ;
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


