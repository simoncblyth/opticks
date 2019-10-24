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

#include "OXPPNS.hh"
#include "OpticksSwitches.h"
#include "plog/Severity.h"

class SLog ; 
class Opticks; 
class OpticksEvent ; 
class OContext ; 
class OBuf ; 

template <typename T> class NPY ; 

/**
OEvent
=======

OptiX buffers representing an OpticksEvent propagation.

The canonical single OEvent instance resides 
in OpEngine and is instanciated with OpEngine.
A pointer is also available in OPropagator, which
is also instanciated with OpEngine.

Buffers are created at the first *upload* and
are subsequently resized to correspond to the OpticksEvent. 

NB upload/download will only act on compute buffers, interop
buffers are skipped within underlying OContext methods
based on OpticksBufferControl settings.


::

    opticks-findl OEvent.hh

    ./okop/OpSeeder.cc
    ./okop/OpEngine.cc
    ./okop/OpIndexerApp.cc
    ./okop/OpZeroer.cc
    ./okop/OpIndexer.cc
    ./okop/tests/OpSeederTest.cc

    ./optixrap/OPropagator.cc
    ./optixrap/OEvent.cc

    ./optixrap/CMakeLists.txt
    ./optixrap/tests/OEventTest.cc
    ./optixrap/oxrap.bash

    ./ok/tests/VizTest.cc



Necessary Buffers
------------------

*genstep*
    (n_genstep,6,4) float32, parameters of Cerenkov, Scintillation or Torch genstep

*photon*
    (n_photon,4,4) float32

*seed*
    (n_photon, 1) uint32, provides genstep_id for each photon  


Buffers During Debugging
-------------------------

*sequence*
    (n_photon, 1, 2) uint64 (unsigned long long) : flag and material sequence (64 bits = 16*4 bits )

*record*
    (n_photon, 16, 2, 4) int16 (shorts)




**/

#include "OXRAP_API_EXPORT.hh"
class OXRAP_API OEvent 
{
    public:
        static const plog::Severity LEVEL ;  
    public:
        enum {
            GENSTEP  = 0x1 << 1, 
            PHOTON   = 0x1 << 2, 
            RECORD   = 0x1 << 3, 
            SEQUENCE = 0x1 << 4,
            SEED     = 0x1 << 5,
            SOURCE   = 0x1 << 6,
            DOWNLOAD_DEFAULT  = PHOTON | RECORD | SEQUENCE 
            };
    public:
        OEvent(Opticks* ok, OContext* ocontext);
        unsigned upload();
        unsigned download();
        void downloadPhotonData();
        unsigned downloadHits();
    private:
        unsigned upload(OpticksEvent* evt);
        unsigned uploadGensteps(OpticksEvent* evt);
#ifdef WITH_SOURCE
        unsigned uploadSource(OpticksEvent* evt);
#endif
        unsigned downloadHitsCompute(OpticksEvent* evt);
        unsigned downloadHitsInterop(OpticksEvent* evt);
    public:
        OContext*     getOContext();
        OpticksEvent* getEvent();
        OBuf* getSeedBuf();
        OBuf* getPhotonBuf();
#ifdef WITH_SOURCE
        OBuf* getSourceBuf();
#endif
        OBuf* getGenstepBuf();
#ifdef WITH_RECORD
        OBuf* getSequenceBuf();
        OBuf* getRecordBuf();
#endif
        void markDirty();
    private:
        void init(); 
        void createBuffers(OpticksEvent* evt);
        void resizeBuffers(OpticksEvent* evt);
        void setEvent(OpticksEvent* evt);
        void download(OpticksEvent* evt, unsigned mask=DOWNLOAD_DEFAULT );
    private:
        SLog*           m_log ; 
        Opticks*        m_ok ; 
        unsigned        m_hitmask ;  
        bool            m_compute ;  
        bool            m_dbghit ; 
        bool            m_dbgdownload ; 
        NPY<unsigned>*  m_mask ; 
        OContext*       m_ocontext ; 
        optix::Context  m_context ; 
        OpticksEvent*   m_evt ; 
        bool            m_photonMarkDirty ; 
#ifdef WITH_SOURCE
        bool            m_sourceMarkDirty ; 
#endif
        bool            m_seedMarkDirty ; 
    protected:
        optix::Buffer   m_genstep_buffer ; 
        optix::Buffer   m_photon_buffer ; 
#ifdef WITH_SOURCE
        optix::Buffer   m_source_buffer ; 
#endif
#ifdef WITH_RECORD
        optix::Buffer   m_record_buffer ; 
        optix::Buffer   m_sequence_buffer ; 
#endif
        optix::Buffer   m_seed_buffer ; 
    private:
        OBuf*           m_genstep_buf ;
        OBuf*           m_photon_buf ;
#ifdef WITH_SOURCE
        OBuf*           m_source_buf ;
#endif
#ifdef WITH_RECORD
        OBuf*           m_record_buf ;
        OBuf*           m_sequence_buf ;
#endif
        OBuf*           m_seed_buf ;
    private:
        bool            m_buffers_created ; 

};


