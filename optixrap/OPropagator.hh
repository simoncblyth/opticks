#pragma once


struct STimes ; 
class SLog ; 
template <typename T> class NPY ;

class OpticksEvent ; 
class OpticksEntry ; 
class Opticks ; 
template <typename T> class OpticksCfg ;
class OpticksHub ; 

#include "OXPPNS.hh"
class cuRANDWrapper ; 

class OContext ; 
class OEvent ; 


#include "OXRAP_API_EXPORT.hh"
class OXRAP_API OPropagator {
    public:
        OPropagator( OpticksHub* hub, OEvent* oevt, OpticksEntry* entry); 
    public:
        void prelaunch();   
        void launch();
    public:
        void setOverride(unsigned int override);
    private:
        void init();
        void setEntry(unsigned int entry);
        void initParameters();
        void initRng();
        void setSize(unsigned width, unsigned height);
    private:
        SLog*                m_log ; 
        OpticksHub*          m_hub ; 
        OEvent*              m_oevt ; 
        OContext*            m_ocontext ; 
        Opticks*             m_ok ; 
        OpticksCfg<Opticks>* m_cfg ; 
        int                  m_override ; 

        OpticksEntry*        m_entry ; 
        int                  m_entry_index ; 

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


