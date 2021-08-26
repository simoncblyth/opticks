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
#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"
#include "plog/Severity.h"

class G4StepPoint ;

// npy-
template <typename T> class NPY ;

// okc-
class Opticks ; 
class OpticksEvent ; 

struct CCtx ; 
struct CPhoton ; 

/**
CWriter
=========

Canonical m_writer instance is resident of CRecorder and is instanciated with it.
Only CRecorder includes this header. 

Writes G4StepPoint to buffer, ie writes step records, final photons and sequence(aka history) entries 
collected from Geant4 into buffers in the "g4evt" OpticksEvent.

In static mode (when gensteps are available ahead of time) the number of photons is known in advance, 
in dynamic mode the buffers are grown as new photons are added.

**/

class CFG4_API CWriter 
{
        static const plog::Severity LEVEL ; 
        friend class CRecorder ; 
    public: 
        // TODO: move into sysrap- 

         static short shortnorm( float v, float center, float extent );
         static unsigned char my__float2uint_rn( float f );

    public:
        CWriter(CCtx& ctx, CPhoton& photon);        
        void setEnabled(bool enabled);
        std::string desc(const char* msg=nullptr) const ; 
        std::string dbgdesc() const ; 
    public:
        // *initEvent*
        //     configures recording and prepares buffer pointers
        //
        void initEvent(OpticksEvent* evt);
        //
        // *writeStepPoint* 
        //     writes the ox,ph,rx buffers emulating Opticks on GPU
        //     invoked by CRecorder::WriteStepPoint
        //
        bool writeStepPoint(const G4StepPoint* point, unsigned flag, unsigned material, bool last );

   private:
        // expand arrays to hold extra gs_photons 
        unsigned expand(unsigned gs_photons);

        // *writeStepPoint_* 
        //     writes the compressed records buffer (rx) 
        //
        void writeStepPoint_(const G4StepPoint* point, const CPhoton& photon, unsigned record_id );

        // *writePhoton* 
        //     writes the photon buffer (ox) and history buffer (ph) aka:seqhis/seqmat
        //     this overwrites prior entries for REJOIN updating record_id 
        //     with dynamic running this means MUST SetTrackSecondariesFirst IN C+S processes (TODO: verify this)
        //
        void writePhoton_(const G4StepPoint* point, unsigned record_id );
        void writeHistory_(unsigned record_id) ;

   private:
        void BeginOfGenstep();

    private:

        CPhoton&           m_photon ; 
        CCtx&              m_ctx ; 
        Opticks*           m_ok ; 
        bool               m_enabled ; 
        OpticksEvent*      m_evt ; 
        unsigned           m_ni ; 

        NPY<short>*               m_records_buffer ; 
        NPY<double>*              m_deluxe_buffer ; 
        NPY<float>*               m_photons_buffer ; 
        NPY<unsigned long long>*  m_history_buffer ; 
};

#include "CFG4_TAIL.hh"


