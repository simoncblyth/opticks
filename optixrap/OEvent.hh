#pragma once

#include "OXPPNS.hh"
class OpticksEvent ; 
class OBuf ; 

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
        OEvent(OContext* ocontext, OpticksEvent* evt);
        void upload(OpticksEvent* evt);
        void download(OpticksEvent* evt, unsigned mask=DEFAULT );
    public:
        OBuf* getSequenceBuf();
        OBuf* getPhotonBuf();
        OBuf* getGenstepBuf();
        OBuf* getRecordBuf();
    private:
        void init(OpticksEvent* evt);
    private:
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

};


