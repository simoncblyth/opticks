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
OKG4Mgr
=========

Highest level manager class for full featured 
Opticks running with Geant4 embedded. 

When "--load" option is not used OKG4Mgr holds a CG4 instance.

**/


class SLog ; 
class Opticks ; 
class OpticksHub ; 
class OpticksIdx; 
class OpticksGen ; 
class OpticksRun ; 
class CG4 ; 
class CGenerator ; 
class OpticksViz ; 
class OKPropagator ; 

#include "plog/Severity.h"
#include "OKG4_API_EXPORT.hh"
#include "OKG4_HEAD.hh"

class OKG4_API OKG4Mgr {
       static const plog::Severity LEVEL ; 
   public:
       OKG4Mgr(int argc, char** argv);
       virtual ~OKG4Mgr();
  private:  
       int preinit() const ;  
       void init() const ;  
  public:
       void propagate();
       void visualize();
       int rc() const ;
   private:
       void propagate_();
       void cleanup();
   private:
       SLog*          m_log ; 
       Opticks*       m_ok ; 
       int            m_preinit ; 
       OpticksRun*    m_run ; 
       OpticksHub*    m_hub ; 
       bool           m_load ; 
       bool           m_nog4propagate ; 
       bool           m_production ; 
       OpticksIdx*    m_idx ; 
       int            m_num_event ; 
       OpticksGen*    m_gen ; 
       CG4*           m_g4 ; 
       CGenerator*    m_generator ; 
       OpticksViz*    m_viz ; 
       OKPropagator*  m_propagator ; 
    
};

#include "OKG4_TAIL.hh"

