#pragma once

#include <string>

struct STimes ; 
class SLog ; 
template <typename T> class NPY ;

class OpticksEvent ; 
class OpticksEntry ; 
class Opticks ; 
template <typename T> class OpticksCfg ;
class OpticksHub ; 

#include "OXPPNS.hh"

class OContext ; 
class ORng ; 
class OEvent ; 

/**
OPropagator
=============

Launch control 


**/


#include "OXRAP_API_EXPORT.hh"
class OXRAP_API OPropagator {
    public:
        OPropagator( OpticksHub* hub, OEvent* oevt, OpticksEntry* entry); 
    public:
        void prelaunch();   
        void launch();
        std::string brief();
    public:
        void setOverride(unsigned int override);
        void setNoPropagate(bool nopropagate=true );
    private:
        void init();
        void setEntry(unsigned int entry);
        void initParameters();
        void setSize(unsigned width, unsigned height);
    private:
        SLog*                m_log ; 
        OpticksHub*          m_hub ; 
        OEvent*              m_oevt ; 
        OContext*            m_ocontext ; 
        optix::Context       m_context ;
        Opticks*             m_ok ; 
        OpticksCfg<Opticks>* m_cfg ; 
        ORng*                m_orng ; 

        int                  m_override ; 
        bool                 m_nopropagate ; 

        OpticksEntry*        m_entry ; 
        int                  m_entry_index ; 

        bool                 m_prelaunch ;

    private:
        unsigned int     m_count ; 
        unsigned int     m_width ; 
        unsigned int     m_height ; 

 
};


