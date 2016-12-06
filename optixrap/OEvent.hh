#pragma once

#include "OXPPNS.hh"
#include "OpticksSwitches.h"

class SLog ; 
class Opticks; 
class OpticksEvent ; 
class OpticksHub ; 
class OContext ; 
class OBuf ; 

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
    (n_photon, 1, 2) uint64 : flag and material sequence (64 bits = 16*4 bits )

*record*
    (n_photon, 16, 2, 4) int16




**/

#include "OXRAP_API_EXPORT.hh"
class OXRAP_API OEvent 
{
    public:
        enum {
            GENSTEP  = 0x1 << 1, 
            PHOTON   = 0x1 << 2, 
            RECORD   = 0x1 << 3, 
            SEQUENCE = 0x1 << 4,
            SEED     = 0x1 << 5,
            DOWNLOAD_DEFAULT  = PHOTON | RECORD | SEQUENCE 
            };
    public:
        OEvent(OpticksHub* hub, OContext* ocontext);
        unsigned upload();
        unsigned download();
        void downloadPhotonData();
        unsigned downloadHits();
    private:
        unsigned upload(OpticksEvent* evt);
        unsigned uploadGensteps(OpticksEvent* evt);
        unsigned downloadHits(OpticksEvent* evt);
    public:
        OContext*     getOContext();
        OpticksEvent* getEvent();
        OBuf* getSeedBuf();
        OBuf* getPhotonBuf();
        OBuf* getGenstepBuf();
#ifdef WITH_RECORD
        OBuf* getSequenceBuf();
        OBuf* getRecordBuf();
#endif
        void markDirty();
    private:
        void createBuffers(OpticksEvent* evt);
        void resizeBuffers(OpticksEvent* evt);
        void setEvent(OpticksEvent* evt);
        void download(OpticksEvent* evt, unsigned mask=DOWNLOAD_DEFAULT );
    private:
        SLog*           m_log ; 
        OpticksHub*     m_hub ; 
        Opticks*        m_ok ; 
        OContext*       m_ocontext ; 
        optix::Context  m_context ; 
        OpticksEvent*   m_evt ; 
        bool            m_photonMarkDirty ; 
        bool            m_seedMarkDirty ; 
    protected:
        optix::Buffer   m_genstep_buffer ; 
        optix::Buffer   m_photon_buffer ; 
#ifdef WITH_RECORD
        optix::Buffer   m_record_buffer ; 
        optix::Buffer   m_sequence_buffer ; 
#endif
        optix::Buffer   m_seed_buffer ; 
    private:
        OBuf*           m_genstep_buf ;
        OBuf*           m_photon_buf ;
#ifdef WITH_RECORD
        OBuf*           m_record_buf ;
        OBuf*           m_sequence_buf ;
#endif
        OBuf*           m_seed_buf ;
    private:
        bool            m_buffers_created ; 

};


