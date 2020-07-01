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

class SLog ; 
class Opticks ; 
class OpticksHub ; 
class OpticksGen ; 
class OpticksRun ; 
class OpticksIdx; 
class OpticksEvent; 
class OpEvt ;

template <typename T> class NPY ; 

class OpPropagator ; 


#include "plog/Severity.h"
#include "OKOP_API_EXPORT.hh"
#include "OKOP_HEAD.hh"

/**
OpMgr : high level steering for compute only Opticks
======================================================

Only used from::

    okop/tests/OpSnapTest
    g4ok/G4Opticks

Canonical OpMgr instance m_opmgr resides in G4Opticks and 
is intanciated by G4Opticks::setGeometry.  Mainly used 
from G4Opticks::propagateOpticalPhotons.

Instanciation creates: OpticksHub, OpticksIdx and OpPropagator.
OpticksHub instanciation will adopt the preexisting GGeo instance
in direct running.

Responsibilities:

1. receive gensteps
2. invoke lower level OpPropagator
3. supply hits 


DevNote
--------

Notice in propagate() repetition of the interplay between 
OpPropagator.m_propagator and OpticksRun.m_run ... 
perhaps factor out into OpKernel ?  

**/


class OKOP_API OpMgr {
   public:
       static const plog::Severity LEVEL ; 
   public:
       OpMgr(Opticks* ok );
       virtual ~OpMgr();
   public:

       void setGensteps(NPY<float>* gensteps); 
       void propagate();
       OpticksEvent* getEvent() const ; 
       OpticksEvent* getG4Event() const ; 
       void reset();

       void snap(const char* dir);
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
       OpPropagator*  m_propagator ; 
       int            m_count ;  

       NPY<float>*    m_gensteps ; 
       NPY<float>*    m_hits ; 
       
};

#include "OKOP_TAIL.hh"

