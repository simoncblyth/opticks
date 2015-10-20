#pragma once

class Composition ; 

class RayTraceConfig ; 
class GGeo ; 
class GMergedMesh ;
class GBoundaryLib ;
class NumpyEvt ; 
class Timer ; 
class cuRANDWrapper ; 
class OBuf ; 
class OGeo ; 
class OFrame ; 
class OBoundaryLib ; 

#include "OContext.hh"
#include "OTimes.hh"

template <typename T> class NPY ;

#include "NPYBase.hpp"
#include "string.h"
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_aabb_namespace.h>

// TODO: split off non-OpenGL OptiXCore for headless usage and easier testing
// TODO: this monolith needs to be drawn and quartered, doing far too much for one class


class OEngine {

    public:

        enum { 
                e_center_extent, 
                e_time_domain, 
                e_wavelength_domain,
                e_number_domain
             } ;

        enum { 
                e_config_idomain,
                e_number_idomain
             } ;

        typedef enum { COMPUTE, INTEROP } Mode_t ;   
        static const char* COMPUTE_ ; 
        static const char* INTEROP_ ; 
    public:
        OEngine(OContext* context, optix::Group top, Mode_t mode=INTEROP );

    public:
        const char* getModeName();
        OEngine::Mode_t getMode();
        bool isCompute();
        bool isInterop();
    public:
        void setOGeo(OGeo* ogeo);
        void setOBoundaryLib(OBoundaryLib* olib);
    public:
        void setComposition(Composition* composition);
        void setGGeo(GGeo* ggeo);

        void setMergedMesh(GMergedMesh* mergedmesh);
        void setBoundaryLib(GBoundaryLib* boundarylib);
        void setEnabled(bool enabled);
        void setOverride(unsigned int override);
        void setDebugPhoton(unsigned int debug_photon);
        void setTrivial(bool trivial);
        void setNumpyEvt(NumpyEvt* evt);
    public:
        void setSize(unsigned int width, unsigned int height);
        void setResolutionScale(unsigned int resolution_scale);
        unsigned int getResolutionScale();
    public:
        void downloadEvt();
    public:
        bool isEnabled();
        void report(const char* msg="OEngine::report");
    public:
        void setRngMax(unsigned int rng_max);
        void setBounceMax(unsigned int bounce_max);
        void setRecordMax(unsigned int record_max);
    public:
        GGeo*         getGGeo();
        GMergedMesh*  getMergedMesh();
        GBoundaryLib* getBoundaryLib();
        optix::Context& getContext();
        optix::Group&   getTopGroup();
        NPY<float>* getDomain();
        NPY<int>*   getIDomain();
    public:
        unsigned int getRngMax();
        unsigned int getBounceMax();
        unsigned int getRecordMax();
        unsigned int getTraceCount();
        unsigned int getDebugPhoton();
    public:
        void init();
        void trace(); 
        void generate();
        void cleanUp();

    public:
        OBuf* getSequenceBuf();
        OBuf* getPhotonBuf();

    private:
        void initRayTrace();
        void initGeometry();
        void initRng();
        void initGenerateOnce();
        void initGenerate();
        void initGenerate(NumpyEvt* evt);

        void preprocess();

        void displayFrame(unsigned int texID);


    public:
        RTformat getFormat(NPYBase::Type_t type);

        template<typename T>
        optix::Buffer   createIOBuffer(NPY<T>* npy, const char* name);

        template <typename T>
        void upload(optix::Buffer& buffer, NPY<T>* npy);

        template <typename T>
        void download(optix::Buffer& buffer, NPY<T>* npy);


    protected:
        optix::Buffer         m_rng_states ;
        unsigned int          m_rng_max ; 
        cuRANDWrapper*        m_rng_wrapper ;


    protected:
        OContext*             m_ocontext; 
        optix::Context        m_context; 
        optix::Buffer         m_genstep_buffer ; 
        optix::Buffer         m_photon_buffer ; 
        optix::Buffer         m_record_buffer ; 
        optix::Buffer         m_sequence_buffer ; 
        optix::Buffer         m_touch_buffer ; 
        optix::Buffer         m_aux_buffer ; 
        optix::Group          m_top ;

        OBuf*                 m_photon_buf ;
        OBuf*                 m_sequence_buf ;

        unsigned int          m_width ;
        unsigned int          m_height ;
        unsigned int          m_resolution_scale ;
        unsigned int          m_photon_buffer_id ;
        unsigned int          m_genstep_buffer_id ;

        Composition*     m_composition ; 

        RayTraceConfig*  m_config ; 
        GGeo*            m_ggeo ; 
        GMergedMesh*     m_mergedmesh ; 
        GBoundaryLib*    m_boundarylib ; 

        OFrame*          m_frame ; 
        OGeo*            m_ogeo ; 
        OBoundaryLib*    m_oboundarylib ; 

        OTimes*          m_trace_times ; 
        unsigned int     m_trace_count ; 
        double           m_trace_prep ; 
        double           m_trace_time ; 

        OTimes*          m_prep_times ; 

        unsigned int     m_generate_count ; 
        unsigned int     m_bounce_max ; 
        unsigned int     m_record_max ; 
        Mode_t           m_mode ; 
        bool             m_enabled ; 
        bool             m_trivial ; 


        int              m_override ; 
        int              m_debug_photon ; 
        NumpyEvt*        m_evt ; 
        NPY<float>*      m_domain ;
        NPY<int>*        m_idomain ;
        Timer*           m_timer ; 


   // from sutil/MeshScene.h
   public:
        void setFilename(const char* filename);
        void loadAccelCache();
        void saveAccelCache();
        std::string getCacheFileName();

   private:
        std::string   m_filename;
        std::string   m_accel_builder;
        std::string   m_accel_traverser;
        std::string   m_accel_refine;
        bool          m_accel_cache_loaded;
        bool          m_accel_caching_on;

};



inline OEngine::OEngine(OContext* ocontext, optix::Group top, Mode_t mode) :
    m_rng_max(0),
    m_rng_wrapper(NULL),
    m_ocontext(ocontext),
    m_context(ocontext->getContext()),
    m_top(top),

    m_photon_buf(NULL),
    m_sequence_buf(NULL),
    m_resolution_scale(1),
    m_photon_buffer_id(0),
    m_genstep_buffer_id(0),

    m_composition(NULL),

    m_config(NULL),
    m_ggeo(NULL),
    m_mergedmesh(NULL),
    m_boundarylib(NULL),
    m_frame(NULL),
    m_ogeo(NULL),
    m_oboundarylib(NULL),

    m_trace_times(new OTimes),
    m_trace_count(0),
    m_trace_prep(0),
    m_trace_time(0),

    m_prep_times(new OTimes),

    m_generate_count(0),
    m_bounce_max(1),
    m_record_max(10),
    m_mode(mode),  
    m_enabled(true),
    m_trivial(false),
    m_override(0),
    m_debug_photon(0),
    m_evt(NULL),
    m_domain(NULL),
    m_idomain(NULL),
    m_timer(NULL),
    m_filename(),
    m_accel_cache_loaded(false),
    m_accel_caching_on(false)
{
}



inline optix::Context& OEngine::getContext()
{
    return m_context ; 
}

inline optix::Group& OEngine::getTopGroup()
{
    return m_top ; 
}


inline void OEngine::setNumpyEvt(NumpyEvt* evt)
{
    m_evt = evt ;
}

inline void OEngine::setComposition(Composition* composition)
{
    m_composition = composition ; 
}
inline void OEngine::setGGeo(GGeo* ggeo)
{
    m_ggeo = ggeo ;
}
inline void OEngine::setOGeo(OGeo* ogeo)
{
    m_ogeo = ogeo ;
}
inline void OEngine::setOBoundaryLib(OBoundaryLib* oboundarylib)
{
    m_oboundarylib = oboundarylib ;
}








inline void OEngine::setFilename(const char* filename)
{
    m_filename = filename ;
}
inline void OEngine::setMergedMesh(GMergedMesh* mergedmesh)
{
    m_mergedmesh = mergedmesh ;
}
inline void OEngine::setBoundaryLib(GBoundaryLib* boundarylib)
{
    m_boundarylib = boundarylib ;
}


inline void OEngine::setEnabled(bool enabled)
{
    m_enabled = enabled ; 
}
inline void OEngine::setTrivial(bool trivial)
{
    m_trivial = trivial ; 
}



inline bool OEngine::isEnabled()
{
    return m_enabled ; 
}


inline GMergedMesh* OEngine::getMergedMesh()
{
    return m_mergedmesh ; 
}
inline GGeo* OEngine::getGGeo()
{
    return m_ggeo ; 
}
inline GBoundaryLib* OEngine::getBoundaryLib()
{
    return m_boundarylib  ; 
}





inline void OEngine::setRngMax(unsigned int rng_max)
{
// default of 0 disables Rng 
// otherwise maximum number of RNG streams, 
// should be a little more than the max number of photons to generate/propagate eg 3e6
    m_rng_max = rng_max ;
}
inline unsigned int OEngine::getRngMax()
{
    return m_rng_max ; 
}

inline void OEngine::setBounceMax(unsigned int bounce_max)
{
    m_bounce_max = bounce_max ;
}
inline unsigned int OEngine::getBounceMax()
{
    return m_bounce_max ; 
}

inline void OEngine::setRecordMax(unsigned int record_max)
{
    m_record_max = record_max ;
}
inline unsigned int OEngine::getRecordMax()
{
    return m_record_max ; 
}





inline unsigned int OEngine::getTraceCount()
{
    return m_trace_count ; 
}


inline NPY<float>* OEngine::getDomain()
{
    return m_domain ;
}
inline NPY<int>* OEngine::getIDomain()
{
    return m_idomain ;
}


inline OBuf* OEngine::getSequenceBuf()
{
    return m_sequence_buf ; 
}
inline OBuf* OEngine::getPhotonBuf()
{
    return m_photon_buf ; 
}


inline OEngine::Mode_t OEngine::getMode()
{
    return m_mode ; 
}
inline bool OEngine::isCompute()
{
    return m_mode == COMPUTE ; 
}
inline bool OEngine::isInterop()
{
    return m_mode == INTEROP ; 
}


inline void OEngine::setOverride(unsigned int override)
{
    m_override = override ; 
}
inline void OEngine::setDebugPhoton(unsigned int debug_photon)
{
    m_debug_photon = debug_photon ; 
}

inline void OEngine::setResolutionScale(unsigned int resolution_scale)
{
    m_resolution_scale = resolution_scale ; 
}
inline unsigned int OEngine::getResolutionScale()
{
    return m_resolution_scale ; 
}



