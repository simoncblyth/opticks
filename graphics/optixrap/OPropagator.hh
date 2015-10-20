#pragma once

#include <optixu/optixpp_namespace.h>
template <typename T> class NPY ;

class cuRANDWrapper ; 
class NumpyEvt ; 
class Composition ; 
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
                e_wavelength_domain,
                e_number_domain
             } ;
    public:
        OPropagator(OContext* ocontext, Composition* composition); 
        // hmm Composition mainly graphical but needed for domains, maybe split these off ?
        void initRng();
    public:
        void initEvent();
        void propagate();
        void downloadEvent();
    public:
        void setNumpyEvt(NumpyEvt* evt);
        void setRngMax(unsigned int rng_max);
        void setBounceMax(unsigned int bounce_max);
        void setRecordMax(unsigned int record_max);

    public:
        NumpyEvt*    getNumpyEvt();
        unsigned int getRngMax();
        unsigned int getBounceMax();
        unsigned int getRecordMax();

    public:
        void setTrivial(bool trivial=true);
        void setOverride(unsigned int override);
    public:
        OBuf* getSequenceBuf();
        OBuf* getPhotonBuf();

    public:
        NPY<float>* getDomain();
        NPY<int>*   getIDomain();

    private:
        void init();
        void initEvent(NumpyEvt* evt);
        void makeDomains();
        void recordDomains();

    private:
        OContext*        m_ocontext ; 
        Composition*     m_composition ; 
        optix::Context   m_context ;
        NumpyEvt*        m_evt ; 
        unsigned int     m_rng_max ; 
        unsigned int     m_bounce_max ; 
        unsigned int     m_record_max ; 

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
        NPY<float>*     m_domain ;
        NPY<int>*       m_idomain ;
 
};



inline OPropagator::OPropagator(OContext* ocontext, Composition* composition) :
    m_ocontext(ocontext),
    m_composition(composition),
    m_evt(NULL),
    m_rng_max(0),
    m_bounce_max(9),
    m_record_max(10),
    m_photon_buf(NULL),
    m_sequence_buf(NULL),
    m_rng_wrapper(NULL),
    m_trivial(false),
    m_times(NULL),
    m_count(0),
    m_prep(0),
    m_time(0),
    m_override(0),
    m_domain(NULL),
    m_idomain(NULL)
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


inline void OPropagator::setRngMax(unsigned int rng_max)
{
// default of 0 disables Rng 
// otherwise maximum number of RNG streams, 
// should be a little more than the max number of photons to generate/propagate eg 3e6
    m_rng_max = rng_max ;
}
inline unsigned int OPropagator::getRngMax()
{
    return m_rng_max ; 
}

inline void OPropagator::setBounceMax(unsigned int bounce_max)
{
    m_bounce_max = bounce_max ;
}
inline unsigned int OPropagator::getBounceMax()
{
    return m_bounce_max ; 
}

inline void OPropagator::setRecordMax(unsigned int record_max)
{
    m_record_max = record_max ;
}
inline unsigned int OPropagator::getRecordMax()
{
    return m_record_max ; 
}






inline NPY<float>* OPropagator::getDomain()
{
    return m_domain ;
}
inline NPY<int>* OPropagator::getIDomain()
{
    return m_idomain ;
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



