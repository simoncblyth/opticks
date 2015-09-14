#pragma once

class Composition ; 
class Renderer ; 
class Texture ; 
class RayTraceConfig ; 
class GGeo ; 
class GMergedMesh ;
class GBoundaryLib ;
class NumpyEvt ; 
class cuRANDWrapper ; 


template <typename T>
class NPY ;

#include "NPYBase.hpp"
#include "string.h"
#include "Touchable.hh"
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_aabb_namespace.h>

// TODO: split off non-OpenGL OptiXCore for headless usage and easier testing
// TODO: this needs to be drawn and quartered, doing far too much for one class

/*

  TODO:

  Refactor code for populating the constitutent OptiX context  
  into OpticksContext class 


*/

class OEngine : public Touchable {

    public:
        enum { 
               e_pinhole_camera,
               e_generate,
               e_entryPointCount 
            };

        enum {
                e_radiance_ray,
                e_touch_ray,
                e_propagate_ray,
                e_rayTypeCount 
             };

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
        unsigned int getNumEntryPoint();
        OEngine(Mode_t mode=INTEROP );

    public:
        const char* getModeName();
        OEngine::Mode_t getMode();
        bool isCompute();
        bool isInterop();
    public:
        void setSize(unsigned int width, unsigned int height);
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
        void downloadEvt();
    public:
        bool isEnabled();
    public:
        void setRngMax(unsigned int rng_max);
        void setBounceMax(unsigned int bounce_max);
        void setRecordMax(unsigned int record_max);
    public:
        GGeo*         getGGeo();
        GMergedMesh*  getMergedMesh();
        GBoundaryLib* getBoundaryLib();
        optix::Context& getContext();
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
        void initRenderer(const char* dir, const char* incl_path);
        void trace(); 
        void generate();
        void render();
        void cleanUp();

    public:
        optix::Buffer& getSequenceBuffer();
        optix::Buffer& getPhoselBuffer();
        optix::Buffer& getRecselBuffer();

    public:
       // fulfil Touchable interface
       unsigned int touch(int ix, int iy);

    private:
        void initRayTrace();
        void initGeometry();
        void initGenerate();
        void initGenerate(NumpyEvt* evt);
        void initRng();
        void preprocess();

        void fill_PBO();
        void displayFrame(unsigned int texID);

    private: 
        void push_PBO_to_Texture(unsigned int texId);
        void associate_PBO_to_Texture(unsigned int texId);

    protected:
        optix::Buffer createOutputBuffer(RTformat format, unsigned int width, unsigned int height);

        // create GL buffer VBO/PBO first then address it as OptiX buffer with optix::Context::createBufferFromGLBO  
        // format can be RT_FORMAT_USER 
        optix::Buffer createOutputBuffer_VBO(unsigned int& id, RTformat format, unsigned int width, unsigned int height);
        optix::Buffer createOutputBuffer_PBO(unsigned int& id, RTformat format, unsigned int width, unsigned int height);


    public:
        RTformat getFormat(NPYBase::Type_t type);

        template<typename T>
        optix::Buffer   createIOBuffer(NPY<T>* npy);

        template <typename T>
        void upload(optix::Buffer& buffer, NPY<T>* npy);

        template <typename T>
        void download(optix::Buffer& buffer, NPY<T>* npy);


    protected:
        optix::Buffer         m_rng_states ;
        unsigned int          m_rng_max ; 
        cuRANDWrapper*        m_rng_wrapper ;


    protected:
        optix::Context        m_context; 
        optix::Buffer         m_output_buffer ; 
        optix::Buffer         m_genstep_buffer ; 
        optix::Buffer         m_photon_buffer ; 
        optix::Buffer         m_record_buffer ; 
        optix::Buffer         m_sequence_buffer ; 
        optix::Buffer         m_touch_buffer ; 
        optix::Buffer         m_phosel_buffer ; 
        optix::Buffer         m_recsel_buffer ; 
        optix::Group          m_top ;

        unsigned int          m_width ;
        unsigned int          m_height ;
        unsigned int          m_photon_buffer_id ;
        unsigned int          m_genstep_buffer_id ;
        unsigned int          m_pbo ;

        unsigned char*   m_pbo_data ; 

        Composition*     m_composition ; 
        Renderer*        m_renderer ; 
        Texture*         m_texture ; 
        RayTraceConfig*  m_config ; 
        GGeo*            m_ggeo ; 
        GMergedMesh*     m_mergedmesh ; 
        GBoundaryLib*    m_boundarylib ; 
        unsigned int     m_trace_count ; 
        unsigned int     m_generate_count ; 
        unsigned int     m_bounce_max ; 
        unsigned int     m_record_max ; 
        Mode_t           m_mode ; 
        bool             m_enabled ; 
        bool             m_trivial ; 
        int              m_texture_id ; 
        int              m_override ; 
        int              m_debug_photon ; 
        NumpyEvt*        m_evt ; 
        NPY<float>*      m_domain ;
        NPY<int>*        m_idomain ;


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



inline OEngine::OEngine(Mode_t mode) :
    m_rng_max(0),
    m_rng_wrapper(NULL),
    m_context(NULL),
    m_top(NULL),
    m_photon_buffer_id(0),
    m_genstep_buffer_id(0),
    m_pbo(0),
    m_pbo_data(NULL),
    m_composition(NULL),
    m_renderer(NULL),
    m_texture(NULL),
    m_config(NULL),
    m_ggeo(NULL),
    m_mergedmesh(NULL),
    m_boundarylib(NULL),
    m_trace_count(0),
    m_generate_count(0),
    m_bounce_max(1),
    m_record_max(10),
    m_mode(mode),  
    m_enabled(true),
    m_trivial(false),
    m_texture_id(-1),
    m_override(0),
    m_debug_photon(0),
    m_evt(NULL),
    m_domain(NULL),
    m_idomain(NULL),
    m_filename(),
    m_accel_cache_loaded(false),
    m_accel_caching_on(true)
{
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



inline optix::Buffer& OEngine::getSequenceBuffer()
{
    return m_sequence_buffer ; 
}
inline optix::Buffer& OEngine::getRecselBuffer()
{
    return m_recsel_buffer ; 
}
inline optix::Buffer& OEngine::getPhoselBuffer()
{
    return m_phosel_buffer ; 
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



