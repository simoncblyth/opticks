#pragma once

#include "OXPPNS.hh"
template <typename T> class NPY ;

class cuRANDWrapper ; 
class OpticksHub ; 
class OpticksEvent ; 
class Opticks ; 

class OContext ; 
class OBuf ; 
struct OTimes ; 

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
        OPropagator(OContext* ocontext, OpticksHub* hub, int override_=0); 
        void initRng();
    public:
        void initEvent();      // creates GPU buffers: genstep, photon, record, sequence
        void prelaunch();
        void launch();
        void downloadEvent();
        void downloadPhotonData();
    public:
        void setEntry(unsigned int entry);
    public:
        void setOverride(unsigned int override);
    public:
        OBuf* getSequenceBuf();
        OBuf* getPhotonBuf();
        OBuf* getGenstepBuf();
        OBuf* getRecordBuf();

        OTimes* getPrelaunchTimes();
        OTimes* getLaunchTimes();
        void dumpTimes(const char* msg="OPropagator::dumpTimes");

    private:
        void init();
        void initEvent(OpticksEvent* evt);
        void makeDomains();
        void recordDomains();

    private:
        OContext*        m_ocontext ; 
        OpticksHub*      m_hub ; 
        Opticks*         m_opticks ; 
        optix::Context   m_context ;
        OTimes*          m_prelaunch_times ; 
        OTimes*          m_launch_times ; 
        bool             m_prelaunch ;
        int              m_entry_index ; 

    protected:
        optix::Buffer   m_genstep_buffer ; 
        optix::Buffer   m_photon_buffer ; 
        optix::Buffer   m_record_buffer ; 
        optix::Buffer   m_sequence_buffer ; 
        optix::Buffer   m_touch_buffer ; 
        optix::Buffer   m_aux_buffer ; 

        OBuf*           m_photon_buf ;
        OBuf*           m_sequence_buf ;
        OBuf*           m_genstep_buf ;
        OBuf*           m_record_buf ;

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


