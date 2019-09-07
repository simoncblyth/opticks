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

#include <map>
#include <vector>
#include <string>
#include "plog/Severity.h"

// npy-
class NConfigurable ; 
class NState ; 

// opticks-
class InterpolatedView ; 

/**
Bookmarks
==========

Manages swapping between states persisted into .ini files within a single *Bookmarks* directory
details handled by npy-/NState

Instantiating *Bookmarks* reads all the .ini state files within the directory into NState 
instances held in m_bookmarks std::map<unsigned int, NState*>

Canonical m_bookmarks instance is a resident of OpticksHub and is instanciated
by OpticksHub::configureState(NConfigurable* scene)

The ctor directory argument of the Bookmarks is provided by NState m_state from okc.Opticks



An InterpolatedView moving between Bookmarks can be created, to result in visible 
changes the viewpoint needs to differ between the states. Currently camera only differences
are not interpolated.

NB trackballed changes need to be collapsed into the view... 

**/

#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

class OKCORE_API Bookmarks {
   static const plog::Severity LEVEL ; 
public:
   enum { UNSET = -1, N=10 };
   typedef std::map<unsigned int, NState*>::const_iterator MUSI ; 
public:
   Bookmarks(const char* dir);
   void setState(NState* state);
   void setVerbose(bool verbose=true);
   void create(unsigned int num);
   std::string description(const char* msg="Bk");
   void Summary(const char* msg="Bookmarks::Summary");
public:
   unsigned int getNumBookmarks();
   MUSI begin();
   MUSI end();
public:
   void setInterpolatedViewPeriod(unsigned int ivperiod); 
   void refreshInterpolatedView();
   InterpolatedView* getInterpolatedView();
private:
   void init(const char* dir);
   void readdir();
   void readmark(unsigned int num);
   void updateTitle();
   InterpolatedView* makeInterpolatedView();
   int parseName(const std::string& basename);
public:
   // Interactor interface
   void number_key_pressed(unsigned int number, unsigned int modifiers=0);
   void number_key_released(unsigned int number);
   void updateCurrent();
public:
   bool exists(unsigned int num);
   unsigned int getCurrent(); 
   const char* getTitle(); 
   void setCurrent(unsigned int num); 

   void collect();  // update state and persist to current slot, writing eg 001.ini
   void apply();    // instruct m_state to apply config setting to associated objects 
public:
   int* getIVPeriodPtr();
   int* getCurrentGuiPtr();
   int* getCurrentPtr();
private:
   const char*                          m_dir ; 
   char                                 m_title[N+1] ;
   NState*                              m_state ; 
   InterpolatedView*                    m_view ;  
   int                                  m_current ; 
   int                                  m_current_gui ; 
   std::map<unsigned int, NState*>      m_bookmarks ;  
   bool                                 m_verbose ; 
   int                                  m_ivperiod ; 

};


#include "OKCORE_TAIL.hh"

