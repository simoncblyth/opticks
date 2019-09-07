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


#include <string>
#include <map>
#include <vector>

#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"

#include "BLocSeqDigest.hh"


template <typename T> 
class BRAP_API BLocSeq
{
        static const unsigned MAX_STEP_SEQ ; 
    public:
        BLocSeq(bool skipdupe);

        void add(const char* loc, int record_id, int step_id); 
        void postStep();
        void mark(T marker); 

    public:
        void dump(const char* msg) const ; 
    private:
        void dumpRecordCounts(const char* msg) const ; 
        void dumpStepCounts(const char* msg) const ; 

    private:
        bool                          m_skipdupe ; 
        unsigned                      m_global_flat_count ; 
        unsigned                      m_step_flat_count ; 
        unsigned                      m_count_mismatch; 

        std::map<unsigned, unsigned>  m_record_count ; 
        std::map<unsigned, unsigned>  m_step_count ; 

        BLocSeqDigest<T>              m_seq ; 

        bool                          m_perstep ; 
        BLocSeqDigest<T>*             m_step_seq ; 
        int                           m_last_step1 ; 


};

#include "BRAP_TAIL.hh"

