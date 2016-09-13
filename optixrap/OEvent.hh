#pragma once

#include "OXPPNS.hh"

class SLog ; 
class OpticksEvent ; 
class OpticksHub ; 
class OContext ; 
class OBuf ; 

/**

OEvent
=======

OptiX buffers representing an OpticksEvent propagation.

The canonical single OEvent instance resides 
in OPropagator and is instanciated together with OPropagator.

Buffers are created at the first *upload* and
are subsequently resized to correspond to the OpticksEvent. 

NB upload/download will only act on compute buffers, interop
buffers are skipped within underlying OContext methods
based on OpticksBufferControl settings.


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
            DEFAULT  = PHOTON | RECORD | SEQUENCE
            };
    public:
        OEvent(OpticksHub* hub, OContext* ocontext);
        void upload();
        void download(unsigned mask=DEFAULT );
        void downloadPhotonData();
        void downloadSeedData();
    private:
        void upload(OpticksEvent* evt);
    public:
        OContext*     getOContext();
        OpticksEvent* getEvent();
        OBuf* getSequenceBuf();
        OBuf* getPhotonBuf();
        OBuf* getGenstepBuf();
        OBuf* getRecordBuf();
        OBuf* getSeedBuf();
        void markDirtyPhotonBuffer();
    private:
        void createBuffers(OpticksEvent* evt);
        void resizeBuffers(OpticksEvent* evt);
        void setEvent(OpticksEvent* evt);
        void download(OpticksEvent* evt, unsigned mask=DEFAULT );
    private:
        SLog*           m_log ; 
        OpticksHub*     m_hub ; 
        OContext*       m_ocontext ; 
        optix::Context  m_context ; 
        OpticksEvent*   m_evt ; 
        bool            m_photonMarkDirty ; 
    protected:
        optix::Buffer   m_genstep_buffer ; 
        optix::Buffer   m_photon_buffer ; 
        optix::Buffer   m_record_buffer ; 
        optix::Buffer   m_sequence_buffer ; 
        optix::Buffer   m_seed_buffer ; 
    private:
        OBuf*           m_genstep_buf ;
        OBuf*           m_photon_buf ;
        OBuf*           m_record_buf ;
        OBuf*           m_sequence_buf ;
        OBuf*           m_seed_buf ;
    private:
        bool            m_buffers_created ; 

};


