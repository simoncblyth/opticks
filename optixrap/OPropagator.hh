#pragma once

#include "OXPPNS.hh"
template <typename T> class NPY ;

class SLog ; 

class cuRANDWrapper ; 
class OpticksHub ; 
class OpticksEvent ; 
class Opticks ; 
template <typename T> class OpticksCfg ;

class OContext ; 
class OpticksEntry ; 
class OBuf ; 
class OEvent ; 
struct STimes ; 


#include "OXRAP_API_EXPORT.hh"
class OXRAP_API OPropagator {
    public:
        OPropagator(OContext* ocontext, OpticksHub* hub, OpticksEntry* entry); 
    public:
        void prelaunch();   // done with the zero event
        void uploadEvent();  
        void launch();
        void downloadEvent();
        void downloadPhotonData();
    public:
        void setOverride(unsigned int override);
    public:
        OBuf* getSequenceBuf();
        OBuf* getPhotonBuf();
        OBuf* getGenstepBuf();
        OBuf* getRecordBuf();
    private:
        void init();
        void setEntry(unsigned int entry);
        void initParameters();
        void initRng();
    private:
        SLog*                m_log ; 
        OContext*            m_ocontext ; 
        OpticksHub*          m_hub ; 
        Opticks*             m_ok ; 
        OpticksCfg<Opticks>* m_cfg ; 
        int                  m_override ; 

        OpticksEntry*        m_entry ; 
        int                  m_entry_index ; 

        OEvent*              m_oevt ; 
        optix::Context       m_context ;
        bool                 m_prelaunch ;

    protected:
        optix::Buffer        m_rng_states ;
        cuRANDWrapper*       m_rng_wrapper ;

    private:
        unsigned int     m_count ; 
        unsigned int     m_width ; 
        unsigned int     m_height ; 

 
};


