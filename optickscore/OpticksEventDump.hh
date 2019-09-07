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


template <typename T> class NPY ; 

#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

class RecordsNPY ; 
class Opticks ;
class OpticksEvent ;
class OpticksEventStat ;
 
class OKCORE_API OpticksEventDump 
{
   public:
       OpticksEventDump( OpticksEvent* evt );
   private:
       void init();
   public:
       void Summary(const char* msg="OpticksEventDump::Summary") const ;
       void dump(unsigned photon_id) const ;
       unsigned getNumPhotons() const ;
   private:
       void dumpRecords(unsigned photon_id ) const ;
       void dumpPhotonData(unsigned photon_id) const ;
   private:
       Opticks*          m_ok ; 
       OpticksEvent*     m_evt ; 
       OpticksEventStat* m_stat ; 
       bool              m_noload ; 
       RecordsNPY*       m_records ; 
       NPY<float>*       m_photons ; 
       NPY<unsigned long long>* m_seq ;
       unsigned          m_num_photons ; 
};

#include "OKCORE_TAIL.hh"

 
