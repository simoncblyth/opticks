#pragma once

#include "OXPPNS.hh"
template <typename T> class NPY ;

class cuRANDWrapper ; 
class OpticksHub ; 
class OpticksEvent ; 
class Opticks ; 

class OContext ; 
class OBuf ; 
class OEvent ; 
struct STimes ; 

// TODO: maybe split OptiX buffer management into an OEvent ?
//       need experience with multi-event running to see 
//       right way to structure 

#include "OXRAP_API_EXPORT.hh"
class OXRAP_API OPropagator {
    public:
        enum { 
                e_config_idomain,
                e_number_idomain
             } ;
        enum { 
                e_center_extent, 
                e_time_domain, 
                e_boundary_domain,
                e_number_domain
             } ;
    public:
        OPropagator(OContext* ocontext, OpticksHub* hub, unsigned entry, int override_=0); 
    public:
        //bool hasInitEvent();
        void initEvent();  // creates OBuf (genstep, photon, record, sequence) + uploads gensteps in compute, already there in interop
        void prelaunch();
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

        STimes* getPrelaunchTimes();
        STimes* getLaunchTimes();
        void dumpTimes(const char* msg="OPropagator::dumpTimes");

    private:
        void init();
        void setEntry(unsigned int entry);
        void initParameters();
        void initRng();
    private:
        void initEventBuffers(OpticksEvent* evt);
        void updateEventBuffers(OpticksEvent* evt);  // compute mode only testing of buffer updating 
    private:
        OContext*        m_ocontext ; 
        OpticksHub*      m_hub ; 
        Opticks*         m_ok ; 
        OpticksEvent*    m_zero ; 
        OEvent*          m_oevt ; 
        optix::Context   m_context ;
        STimes*          m_prelaunch_times ; 
        STimes*          m_launch_times ; 
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
        //bool            m_init_event ; 
 
};


