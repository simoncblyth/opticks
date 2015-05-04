#pragma once

class Composition ; 
class Renderer ; 
class Texture ; 
class RayTraceConfig ; 
class GGeo ; 
class GMergedMesh ;
class NPY ;
class NumpyEvt ; 

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_aabb_namespace.h>

// TODO: split off non-OpenGL OptiXCore for headless usage and easier testing

class OptiXEngine {

    public:
        enum { 
               e_pinhole_camera,
               e_generate,
               e_entryPointCount 
            };

        enum {
                e_radiance_ray,
                e_touch_ray,
                e_rayTypeCount 
             };

    public:
        OptiXEngine(const char* cmake_target);

        void setSize(unsigned int width, unsigned int height);
        void setComposition(Composition* composition);
        void setGGeo(GGeo* ggeo);
        void setMergedMesh(GMergedMesh* mergedmesh);
        void setEnabled(bool enabled);
        void setNumpyEvt(NumpyEvt* evt);

        GMergedMesh* getMergedMesh();

        void init();
        void initRenderer();
        void initContext();
        void initGeometry();
        void initGenerate();
        void initGenerate(NumpyEvt* evt);

        void preprocess(); 
        void trace(); 
        void generate();

        void cleanUp();
        void fill_PBO();
        void displayFrame(unsigned int texID);

    public:
        optix::Context& getContext();
        void render();
        
    private: 
        //unsigned int getPBOId(){ return m_pbo ; }
        //unsigned int getVBOId(){ return m_vbo ; }
        void push_PBO_to_Texture(unsigned int texId);
        void associate_PBO_to_Texture(unsigned int texId);

    protected:
        optix::Buffer createOutputBuffer(RTformat format, unsigned int width, unsigned int height);

        // create GL buffer VBO/PBO first then address it as OptiX buffer with optix::Context::createBufferFromGLBO  
        // format can be RT_FORMAT_USER 
        optix::Buffer createOutputBuffer_VBO(unsigned int& id, RTformat format, unsigned int width, unsigned int height);
        optix::Buffer createOutputBuffer_PBO(unsigned int& id, RTformat format, unsigned int width, unsigned int height);

    protected:
        optix::Context        m_context; 
        optix::Buffer         m_output_buffer ; 
        optix::Buffer         m_genstep_buffer ; 
        optix::Buffer         m_photon_buffer ; 
        optix::GeometryGroup  m_geometry_group ;
        optix::Aabb           m_aabb;

        unsigned int          m_width ;
        unsigned int          m_height ;
        unsigned int          m_photon_buffer_id ;
        unsigned int          m_genstep_buffer_id ;
        unsigned int          m_pbo ;


        unsigned char* m_pbo_data ; 

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
