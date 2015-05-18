#pragma once

class Composition ; 
class Renderer ; 
class Texture ; 
class RayTraceConfig ; 
class GGeo ; 
class GMergedMesh ;
class NPY ;
class NumpyEvt ; 
class cuRANDWrapper ; 

#include "Touchable.hh"
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_aabb_namespace.h>

// TODO: split off non-OpenGL OptiXCore for headless usage and easier testing
// TODO: this needs to be drawn and quartered, doing far too much for one class

class OptiXEngine : public Touchable {

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

    public:
        OptiXEngine(const char* cmake_target);

    public:
        void setSize(unsigned int width, unsigned int height);
        void setComposition(Composition* composition);
        void setGGeo(GGeo* ggeo);
        void setMergedMesh(GMergedMesh* mergedmesh);
        void setEnabled(bool enabled);
        void setNumpyEvt(NumpyEvt* evt);
        void setRngMax(unsigned int rng_max);

    public:
        GMergedMesh* getMergedMesh();
        optix::Context& getContext();
        unsigned int getRngMax();
        unsigned int getTraceCount();

    public:
        void init();
        void trace(); 
        void generate();
        void render();
        void cleanUp();

    public:
       // fulfil Touchable interface
       void touch(unsigned char key, int ix, int iy);

    private:
        void initRenderer();
        void initContext();
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

    protected:
        optix::Buffer         m_rng_states ;
        unsigned int          m_rng_max ; 
        cuRANDWrapper*        m_rng_wrapper ;

    protected:
        optix::Context        m_context; 
        optix::Buffer         m_output_buffer ; 
        optix::Buffer         m_genstep_buffer ; 
        optix::Buffer         m_photon_buffer ; 
        optix::Buffer         m_touch_buffer ; 
        optix::GeometryGroup  m_geometry_group ;
        //optix::Aabb           m_aabb;

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
        unsigned int     m_trace_count ; 
        unsigned int     m_generate_count ; 
        char*            m_cmake_target ;
        bool             m_enabled ; 
        int              m_texture_id ; 
        NumpyEvt*        m_evt ; 


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

inline void OptiXEngine::setNumpyEvt(NumpyEvt* evt)
{
    m_evt = evt ;
}

inline void OptiXEngine::setComposition(Composition* composition)
{
    m_composition = composition ; 
}
inline void OptiXEngine::setGGeo(GGeo* ggeo)
{
    m_ggeo = ggeo ;
}
inline void OptiXEngine::setFilename(const char* filename)
{
    m_filename = filename ;
}
inline void OptiXEngine::setMergedMesh(GMergedMesh* mergedmesh)
{
    m_mergedmesh = mergedmesh ;
}
inline void OptiXEngine::setEnabled(bool enabled)
{
    m_enabled = enabled ; 
}


inline GMergedMesh* OptiXEngine::getMergedMesh()
{
    return m_mergedmesh ; 
}




inline void OptiXEngine::setRngMax(unsigned int rng_max)
{
// default of 0 disables Rng 
// otherwise maximum number of RNG streams, 
// should be a little more than the max number of photons to generate/propagate eg 3e6
    m_rng_max = rng_max ;
}
inline unsigned int OptiXEngine::getRngMax()
{
    return m_rng_max ; 
}
inline unsigned int OptiXEngine::getTraceCount()
{
    return m_trace_count ; 
}






