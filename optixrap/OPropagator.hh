#pragma once

#include "OXPPNS.hh"
template <typename T> class NPY ;

class SLog ; 

class cuRANDWrapper ; 
class OpticksHub ; 
class OpticksEvent ; 
class Opticks ; 

class OContext ; 
class OBuf ; 
class OEvent ; 
struct STimes ; 


#include "OXRAP_API_EXPORT.hh"
class OXRAP_API OPropagator {
    public:
        static OPropagator* make(OContext* ocontext, OpticksHub* hub );
        OPropagator(OContext* ocontext, OpticksHub* hub, unsigned entry, int override_=0); 
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
        SLog*            m_log ; 
        OContext*        m_ocontext ; 
        OpticksHub*      m_hub ; 
        Opticks*         m_ok ; 
        OEvent*          m_oevt ; 
        optix::Context   m_context ;
        bool             m_prelaunch ;
        int              m_entry_index ; 

    protected:
        optix::Buffer   m_touch_buffer ; 
        optix::Buffer   m_aux_buffer ; 

    protected:
        optix::Buffer   m_rng_states ;
        cuRANDWrapper*  m_rng_wrapper ;

    private:
        bool             m_trivial ; 
        unsigned int     m_count ; 
        unsigned int     m_width ; 
        unsigned int     m_height ; 
        double           m_prep ; 
        double           m_time ; 

    private:
        int             m_override ; 
 
};


