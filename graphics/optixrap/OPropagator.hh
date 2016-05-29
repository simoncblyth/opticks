#pragma once

#include <optixu/optixpp_namespace.h>
template <typename T> class NPY ;

class cuRANDWrapper ; 
class NumpyEvt ; 
class Opticks ; 

class OContext ; 
class OBuf ; 
struct OTimes ; 

class OPropagator {
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
        OPropagator(OContext* ocontext, Opticks* opticks); 
        void initRng();
    public:
        void initEvent();      // creates GPU buffers: genstep, photon, record, sequence
        void prelaunch();
        void launch();
        void downloadEvent();
    public:
        void setNumpyEvt(NumpyEvt* evt);
        NumpyEvt*    getNumpyEvt();

    public:
        void setTrivial(bool trivial=true);
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
        void initEvent(NumpyEvt* evt);
        void makeDomains();
        void recordDomains();

    private:
        OContext*        m_ocontext ; 
        Opticks*         m_opticks ; 
        optix::Context   m_context ;
        NumpyEvt*        m_evt ; 
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



inline OPropagator::OPropagator(OContext* ocontext, Opticks* opticks) 
   :
    m_ocontext(ocontext),
    m_opticks(opticks),
    m_evt(NULL),
    m_prelaunch_times(NULL),
    m_launch_times(NULL),
    m_prelaunch(false),
    m_entry_index(-1),
    m_photon_buf(NULL),
    m_sequence_buf(NULL),
    m_genstep_buf(NULL),
    m_record_buf(NULL),
    m_rng_wrapper(NULL),
    m_trivial(false),
    m_count(0),
    m_width(0),
    m_height(0),
    m_prep(0),
    m_time(0),
    m_override(0)
{
    init();
}

inline void OPropagator::setTrivial(bool trivial)
{
    m_trivial = trivial ; 
}
inline void OPropagator::setOverride(unsigned int override)
{
    m_override = override ; 
}



inline void OPropagator::setNumpyEvt(NumpyEvt* evt)
{
    m_evt = evt ;
}
inline NumpyEvt* OPropagator::getNumpyEvt()
{
    return m_evt ;
}

inline OBuf* OPropagator::getSequenceBuf()
{
    return m_sequence_buf ; 
}
inline OBuf* OPropagator::getPhotonBuf()
{
    return m_photon_buf ; 
}
inline OBuf* OPropagator::getGenstepBuf()
{
    return m_genstep_buf ; 
}
inline OBuf* OPropagator::getRecordBuf()
{
    return m_record_buf ; 
}

inline OTimes* OPropagator::getPrelaunchTimes()
{
    return m_prelaunch_times ; 
}

inline OTimes* OPropagator::getLaunchTimes()
{
    return m_launch_times ; 
}


