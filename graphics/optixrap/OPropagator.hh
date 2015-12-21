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
        void initEvent();
        void propagate();
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

    protected:
        optix::Buffer   m_genstep_buffer ; 
        optix::Buffer   m_photon_buffer ; 
        optix::Buffer   m_record_buffer ; 
        optix::Buffer   m_sequence_buffer ; 
        optix::Buffer   m_touch_buffer ; 
        optix::Buffer   m_aux_buffer ; 

        OBuf*           m_photon_buf ;
        OBuf*           m_sequence_buf ;

    protected:
        optix::Buffer   m_rng_states ;
        cuRANDWrapper*  m_rng_wrapper ;

    private:
        bool             m_trivial ; 
        OTimes*          m_times ; 
        unsigned int     m_count ; 
        double           m_prep ; 
        double           m_time ; 

    private:
        int             m_override ; 
 
};



inline OPropagator::OPropagator(OContext* ocontext, Opticks* opticks) :
    m_ocontext(ocontext),
    m_opticks(opticks),
    m_evt(NULL),
    m_photon_buf(NULL),
    m_sequence_buf(NULL),
    m_rng_wrapper(NULL),
    m_trivial(false),
    m_times(NULL),
    m_count(0),
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
inline void OPropagator::setOverride(unsigned int override)
{
    m_override = override ; 
}



