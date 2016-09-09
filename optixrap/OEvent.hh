#pragma once

#include "OXPPNS.hh"

class SLog ; 
class OpticksEvent ; 
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
            DEFAULT  = PHOTON | RECORD | SEQUENCE
            };
    public:
        OEvent(OContext* ocontext);
        void upload(OpticksEvent* evt);
        void download(unsigned mask=DEFAULT );
    public:
        OpticksEvent* getEvent();
        OBuf* getSequenceBuf();
        OBuf* getPhotonBuf();
        OBuf* getGenstepBuf();
        OBuf* getRecordBuf();
    private:
        void createBuffers(OpticksEvent* evt);
        void resizeBuffers(OpticksEvent* evt);
        void setEvent(OpticksEvent* evt);
        void download(OpticksEvent* evt, unsigned mask=DEFAULT );
    private:
        SLog*           m_log ; 
        OContext*       m_ocontext ; 
        optix::Context  m_context ; 
        OpticksEvent*   m_evt ; 
    protected:
        optix::Buffer   m_genstep_buffer ; 
        optix::Buffer   m_photon_buffer ; 
        optix::Buffer   m_record_buffer ; 
        optix::Buffer   m_sequence_buffer ; 
    private:
        OBuf*           m_genstep_buf ;
        OBuf*           m_photon_buf ;
        OBuf*           m_record_buf ;
        OBuf*           m_sequence_buf ;
        bool            m_buffers_created ; 

};


