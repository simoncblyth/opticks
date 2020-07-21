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

#include "plog/Severity.h"

class SLog ; 
class Opticks ; 
class OpticksHub ; 
class OpticksGen ; 
class OpticksRun ; 
class OpticksIdx; 
class OpticksViz ; 
class OKPropagator ; 

#include "OK_API_EXPORT.hh"
#include "OK_HEAD.hh"

/**
OKMgr
======

Together with OKG4Mgr the highest of high level control.
Used from primary applications such as *OKTest* (ok/tests/OKTest.cc)

**/


class OK_API OKMgr {
   public:
       static const plog::Severity LEVEL ; 
   public:
       OKMgr(int argc, char** argv, const char* argforced=0 );
       virtual ~OKMgr();
   public:
       void propagate();
       void visualize();
       int rc() const ; 
   private:
       void init();
       void cleanup();
   private:
       SLog*          m_log ; 
       Opticks*       m_ok ; 
       OpticksHub*    m_hub ; 
       OpticksIdx*    m_idx ; 
       int            m_num_event ;  
       OpticksGen*    m_gen ; 
       OpticksRun*    m_run ; 
       OpticksViz*    m_viz ; 
       OKPropagator*  m_propagator ; 
       int            m_count ;  
       
};

#include "OK_TAIL.hh"

