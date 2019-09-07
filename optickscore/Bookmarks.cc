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

#include <cstdio>
#include <cstring>
#include <sstream>

#include <boost/lexical_cast.hpp>

// brap-
#include "BFile.hh"
#include "BDir.hh"
#include "PLOG.hh"

// npy-
#include "NGLM.hpp"
#include "NState.hpp"

// optickscore-
#include "OpticksConst.hh"
#include "InterpolatedView.hh"
#include "Bookmarks.hh"


const plog::Severity Bookmarks::LEVEL = PLOG::EnvLevel("Bookmarks", "DEBUG"); 


Bookmarks::Bookmarks(const char* dir)  
       :
       m_dir(NULL),
       m_state(NULL),
       m_view(NULL),
       m_current(UNSET),
       m_current_gui(UNSET),
       m_verbose(false),
       m_ivperiod(100)
{
    init(dir);
}


int* Bookmarks::getIVPeriodPtr()
{
    return &m_ivperiod ; 
}
int* Bookmarks::getCurrentGuiPtr()
{
    return &m_current_gui ; 
}
int* Bookmarks::getCurrentPtr()
{
    return &m_current ; 
}


unsigned int Bookmarks::getNumBookmarks()
{
    return m_bookmarks.size(); 
}
Bookmarks::MUSI Bookmarks::begin()
{
    return m_bookmarks.begin();
}
Bookmarks::MUSI Bookmarks::end()
{
    return m_bookmarks.end();
}



const char* Bookmarks::getTitle()
{
   return &m_title[0] ; 
}

void Bookmarks::setVerbose(bool verbose)
{
   m_verbose = verbose ; 
}
void Bookmarks::setInterpolatedViewPeriod(unsigned int ivperiod)
{
   m_ivperiod = ivperiod ; 
}


bool Bookmarks::exists(unsigned int num)
{
    return m_bookmarks.count(num) == 1 ; 
}

void Bookmarks::setCurrent(unsigned int num)
{
    m_current = num ; 
}
unsigned int Bookmarks::getCurrent()
{
    return m_current ; 
}















void Bookmarks::init(const char* dir)
{
    std::string _dir = BFile::FormPath(dir) ;

    LOG(LEVEL)
        << " dir " << ( dir ? dir : "NULL" )
        << " expandvars dir " << _dir 
        ; 

    m_dir = strdup(_dir.c_str());
    readdir();
}

void Bookmarks::setState(NState* state)
{
    m_state = state ; 
}


std::string Bookmarks::description(const char* msg)
{
    std::stringstream ss ; 
    ss << msg << " " << getTitle() ; 
    return ss.str();
}

void Bookmarks::Summary(const char* msg)
{
    LOG(info) << description(msg);
}


void Bookmarks::updateTitle()
{
    m_title[N] = '\0' ;
    for(char i=0 ; i < N ; i++) *(m_title + i) = exists(i) ? i + '0' : '_' ; 
}


int Bookmarks::parseName(const std::string& basename)
{
    int num(UNSET);
    try
    { 
        num = boost::lexical_cast<int>(basename) ;
    }   
    catch (const boost::bad_lexical_cast& e ) 
    { 
        LOG(warning)  << "Caught bad lexical cast with error " << e.what() ;
    }   
    catch( ... )
    {
        LOG(warning) << "Unknown exception caught!" ;
    }   
    return num ; 
}



void Bookmarks::readdir()
{
    m_bookmarks.clear();
    LOG(debug) << "Bookmarks::readdir " << m_dir ;

    typedef std::vector<std::string> VS ;
    VS basenames ; 
    BDir::dirlist(basenames, m_dir, ".ini" );  // basenames do not include the .ini


    for(VS::const_iterator it=basenames.begin() ; it != basenames.end() ; it++)
    {
        std::string basename = *it ; 
        int num = parseName(basename); 
        if(num == UNSET) continue ; 
        readmark(num);
    }
    updateTitle();
}


void Bookmarks::readmark(unsigned int num)
{
    //if(num == UNSET) return;   will never happn UNSET is -1
    m_bookmarks[num] = NState::load(m_dir, num ) ; 
}


void Bookmarks::number_key_released(unsigned int num)
{
    LOG(debug) << "Bookmarks::number_key_released " << num ; 
}

void Bookmarks::number_key_pressed(unsigned int num, unsigned int modifiers)
{
    LOG(debug) << "Bookmarks::number_key_pressed "
               << " num "  << num 
               << " modifiers " << OpticksConst::describeModifiers(modifiers) 
               ; 

    bool shift = OpticksConst::isShift(modifiers) ;
    bool exists_ = exists(num);
    if(exists_)
    {
        if(m_current != UNSET && int(num) == m_current && shift)
        { 
            // repeating pressing a num key when on that bookmark with shift down
            LOG(info) << "Bookmarks::number_key_pressed repeat current book mark with shift" ;
            m_state->save();
            // the save updates, prior to persisting
        }
        else
        { 
            // set m_state name from m_current, load and apply : ie set values of attached configurables
            setCurrent(num);
            apply();  
        }
    }
    else
    {
        if(OpticksConst::isShift(modifiers))
        {
            create(num);
        }
        else
        {
            LOG(info) << "Bookmarks::number_key_pressed no such bookmark  " << num << " (use shift modifier to create) " ; 
        }
    }
}




void Bookmarks::create(unsigned int num)
{
    setCurrent(num);
    LOG(info) << "Bookmarks::create : persisting state to slot " << m_current ; 
    collect();
    readdir();   // clears and reloads all bookmarks, updating existance/states from the persisted files 
}

void Bookmarks::updateCurrent()
{
    bool exists_ = exists(m_current);
    if(exists_)
    {
        LOG(info) << "Bookmarks::updateCurrent persisting state " ;
        collect();
        readmark(m_current);
    }
    else
    {
        LOG(info) << "Bookmarks::updateCurrent m_current doesnt exist " << m_current ;
    }
}

void Bookmarks::collect()
{
    if(m_current == UNSET ) return ; 

    if(m_verbose) LOG(info) << "Bookmarks::collect " << m_current ; 

    m_state->collect();
    m_state->setName(m_current);
    m_state->save();
}

void Bookmarks::apply()
{
    if(m_current == UNSET ) return ; 
    if(m_verbose) LOG(info) << "Bookmarks::apply " << m_current ; 

    m_state->setName(m_current);
    int rc = m_state->load();
    if(rc == 0)
    {
        m_state->apply();
    } 
    else
    {
        LOG(warning) << "Bookmarks::apply FAILED for m_current " << m_current ; 
        m_current = UNSET ; 
    } 
}

InterpolatedView* Bookmarks::makeInterpolatedView()
{
    if(m_bookmarks.size() < 2)
    {
        LOG(warning) << "Bookmarks::makeInterpolatedView" 
                     << " requires at least 2 bookmarks "
                     ;

        return NULL ; 
    }

    InterpolatedView* iv = new InterpolatedView(m_ivperiod) ; 
    for(MUSI it=m_bookmarks.begin() ; it!=m_bookmarks.end() ; it++)
    {
         NState* state = it->second ; 

         View* v = new View ; 
         state->addConfigurable(v);
         state->apply();

         iv->addView(v);
    }
    return iv ; 
}

void Bookmarks::refreshInterpolatedView()
{
    delete m_view ; 
    m_view = NULL ; 
}

InterpolatedView* Bookmarks::getInterpolatedView()
{
    if(!m_view) m_view = makeInterpolatedView();
    return m_view ;             
}


