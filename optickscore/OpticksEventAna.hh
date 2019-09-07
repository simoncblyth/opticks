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
#include <map>

class NCSGList ; 
struct NCSGIntersect ;
//struct nnode ; 
template <typename T> class NPY ; 
class RecordsNPY ; 
class NGeoTestConfig ; 

class Opticks ; 
class OpticksEvent ; 
class OpticksEventStat ; 

#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

/**
OpticksEventAna
=================

Splaying GScene::anaEvent into reusability.


Ideas
------

Handle multiple SDFs from NCSGList to check all nodes in 
test geometry ... so cwan infer the nodeindex 
from each photon position (excluding bulk positions, SC, AB )


**/



class OKCORE_API OpticksEventAna
{
   public:
        OpticksEventAna( Opticks* ok, OpticksEvent* evt, NCSGList* csglist );

        std::string desc();
        void dump(const char* msg="OpticksEventAna::dump");
        void dumpPointExcursions(const char* msg="OpticksEventAna::dumpPointExcursions");
   private:
        void init();
        void initOverride(NGeoTestConfig* gtc);
        void initSeqMap();

        void countPointExcursions();
        void checkPointExcursions(); // using the seqmap expectations

   private:
        Opticks*           m_ok ; 

        float              m_epsilon ;                      
        unsigned long long m_dbgseqhis ;
        unsigned long long m_dbgseqmat ;


        unsigned long long m_seqmap_his ;
        unsigned long long m_seqmap_val ;
        bool               m_seqmap_has ; 

        unsigned long long m_seqhis_select ;

        OpticksEvent*            m_evt ; 
        NGeoTestConfig*          m_evtgtc ; 
      
        NCSGList*                m_csglist ;
        unsigned                 m_tree_num ; 
        NCSGIntersect*           m_csgi ;  


        OpticksEventStat*        m_stat ; 
        RecordsNPY*              m_records ; 

        NPY<float>*              m_pho  ; 
        NPY<unsigned long long>* m_seq ;
        unsigned                 m_pho_num ; 
        unsigned                 m_seq_num ; 

};


#include "OKCORE_TAIL.hh"

