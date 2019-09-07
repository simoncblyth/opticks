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

//template <typename T> class NPY ; 
//class RecordsNPY ; 

class Opticks ; 
class OpticksEvent ; 
class OpticksEventStat ; 
class OpticksEventDump ; 

#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

/**
OpticksEventCompare
======================

**/



class OKCORE_API OpticksEventCompare
{
   public:
       OpticksEventCompare( OpticksEvent* a, OpticksEvent* b);
       void dump(const char* msg="OpticksEventCompare::dump") const ;
       void dumpMatchedSeqHis() const ;
   private:
       Opticks*                 m_ok ; 
       unsigned long long m_dbgseqhis ;
       unsigned long long m_dbgseqmat ;

       OpticksEvent*            m_a ; 
       OpticksEvent*            m_b ; 
      
       OpticksEventStat*        m_as ; 
       OpticksEventStat*        m_bs ; 
      
       OpticksEventDump*        m_ad ; 
       OpticksEventDump*        m_bd ; 




};


#include "OKCORE_TAIL.hh"

