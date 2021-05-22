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

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"
#include "plog/Severity.h"

class G4StepPoint ;


// npy-
template <typename T> class NPY ;

// okc-
class Opticks ; 
class OpticksEvent ; 

struct CG4Ctx ; 
struct CPhoton ; 

/**
CWriter
=========

Canonical m_writer instance is resident of CRecorder and is instanciated with it.

Writes step records, final photons and sequence(aka history) entries 
collected from Geant4 into buffers in the "g4evt" OpticksEvent.

In static mode the number of photons is known in advance, in dynamic
mode the buffers are grown as new photons are added.

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
        CWriter(CG4Ctx& ctx, CPhoton& photon, bool dynamic);        

        void setEnabled(bool enabled);
        bool writeStepPoint(const G4StepPoint* point, unsigned flag, unsigned material, bool last );
        void writePhoton(const G4StepPoint* point );
        // *writePhoton* overwrites prior entries for REJOIN updating target_record_id 
   private:
        void writeStepPoint_(const G4StepPoint* point, const CPhoton& photon );
        void initEvent(OpticksEvent* evt);
    private:

        CPhoton&           m_photon ; 
        bool               m_dynamic ; 
        CG4Ctx&            m_ctx ; 
        Opticks*           m_ok ; 
        bool               m_enabled ; 

        OpticksEvent*      m_evt ; 

        NPY<float>*               m_primary ; 

        NPY<short>*               m_records_buffer ; 
        NPY<float>*               m_photons_buffer ; 
        NPY<unsigned long long>*  m_history_buffer ; 

        NPY<short>*               m_dynamic_records ; 
        NPY<float>*               m_dynamic_photons ; 
        NPY<unsigned long long>*  m_dynamic_history ; 

        NPY<short>*               m_target_records ; 

        unsigned           m_verbosity ; 


};

#include "CFG4_TAIL.hh"


