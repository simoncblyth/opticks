#pragma once

#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cstring>
#include <string>
#include <vector>

#include <glm/glm.hpp>


class NumpyEvt ; 
class MultiViewNPY ;

// ggeo-
class GDrawable ;
class GMergedMesh ;
class GGeo ;
class GBoundaryLibMetadata ;
class GBuffer ; 

// oglrap-
class DynamicDefine ; 
class Renderer ; 
class Rdr ;
class Device ; 
class Composition ; 
class Photons ; 
class Colors ; 


#include "NConfigurable.hpp"


class Scene : public NConfigurable {
   public:
        static const char* PREFIX ;
        const char* getPrefix();
   public:
        static const char* PHOTON ;
        static const char* AXIS ;
        static const char* GENSTEP ;
        static const char* GLOBAL ;
        static const char* RECORD ;
   public:
        static const char* TARGET ;
   public:
        static const char* REC_ ; 
        static const char* ALTREC_ ; 
        static const char* DEVREC_ ; 
   public:
        static const char* BBOX_ ; 
        static const char* NORM_ ; 
        static const char* NONE_ ; 
        static const char* WIRE_ ; 
        static const char* NORM_BBOX_ ; 
   public:
        enum { MAX_INSTANCE_RENDERER = 5 };  
        static const char* _INSTANCE   ;
        static const char* INSTANCE0  ;
        static const char* INSTANCE1  ;
        static const char* INSTANCE2  ;
        static const char* INSTANCE3  ;
        static const char* INSTANCE4  ;
        static const char* _BBOX   ;
        static const char* BBOX0  ;
        static const char* BBOX1  ;
        static const char* BBOX2  ;
        static const char* BBOX3  ;
        static const char* BBOX4  ;

   public:
        void setRenderMode(const char* s);
        std::string getRenderMode();
   public:
        typedef enum { REC, ALTREC, DEVREC, NUM_RECORD_STYLE } RecordStyle_t ;
        void setRecordStyle(Scene::RecordStyle_t style);
        Scene::RecordStyle_t getRecordStyle();
        static const char* getRecordStyleName(Scene::RecordStyle_t style);
        const char* getRecordStyleName();
        void nextPhotonStyle();
   public:
        // disabled styles after NUM_GEOMETRY_STYLE
        typedef enum { BBOX, NORM, NONE, WIRE, NUM_GEOMETRY_STYLE, NORM_BBOX } GeometryStyle_t ;
        void setGeometryStyle(Scene::GeometryStyle_t style);
        unsigned int getNumGeometryStyle(); // allows ro override the enum
        void setNumGeometryStyle(unsigned int num_geometry_style); // used to disable WIRE style for JUNO
        void applyGeometryStyle();
        static const char* getGeometryStyleName(Scene::GeometryStyle_t style);
        const char* getGeometryStyleName();
        void nextGeometryStyle();
   public:
        void setWireframe(bool wire=true);
   public:
        typedef enum { GVIS, GINVIS, GVISVEC, GVEC, NUM_GLOBAL_STYLE } GlobalStyle_t ;  
        unsigned int getNumGlobalStyle(); 
        void setNumGlobalStyle(unsigned int num_global_style); // used to disable GVISVEC GVEC styles for JUNO
        void nextGlobalStyle();  // key Q (quiet)
        void applyGlobalStyle();
   public:
        typedef enum { R_PROJECTIVE, R_RAYTRACED, R_COMPOSITE,  NUM_RENDER_STYLE } RenderStyle_t ;  
        void nextRenderStyle();  // key O (formerly optix render mode toggle)
        void applyRenderStyle();
        bool isProjectiveRender();
        bool isRaytracedRender();
        bool isCompositeRender();
   public:
        typedef enum { IVIS, IINVIS, NUM_INSTANCE_STYLE } InstanceStyle_t ;  
        void nextInstanceStyle();
        void applyInstanceStyle();
   public:
        Scene(const char* shader_dir=NULL, const char* shader_incl_path=NULL, const char* shader_dynamic_dir=NULL );
   private:
        void init();
   public:
        void write(DynamicDefine* dd);
        void gui();
        void initRenderers();
   public:
        // Configurable
        std::vector<std::string> getTags();
        void set(const char* name, std::string& xyz);
        std::string get(const char* name);

   public:
        static bool accepts(const char* name);
        void configure(const char* name, const char* value_);
        void configure(const char* name, int value);

   public:
        void configureI(const char* name, std::vector<int> values);
        void setComposition(Composition* composition);
        void setNumpyEvt(NumpyEvt* evt);
        void setPhotons(Photons* photons);
   public:
        void setGeometry(GGeo* gg);
        GGeo* getGeometry();
        void setFaceTarget(unsigned int face_index, unsigned int solid_index, unsigned int mesh_index);
        void setFaceRangeTarget(unsigned int face_index0, unsigned int face_index1, unsigned int solid_index, unsigned int mesh_index);
        void uploadGeometry(); 
   public:
        void uploadColorBuffer(GBuffer* colorbuffer);
   public:
        // target cannot live in Composition, as needs geometry 
        // to convert solid index into CenterExtent to give to Composition
        //
        void setTarget(unsigned int index=0, bool aim=true); 
        unsigned int getTarget(); 
        unsigned int getTargetDeferred();

        unsigned int touch(int ix, int iy, float depth);
        void setTouch(unsigned int index); 
        unsigned int getTouch();
        void jump(); 

   public:
        void upload();
        void uploadSelection();
   private:
        void uploadEvt();
        void uploadAxis();
   private:
        void uploadRecordAttr(MultiViewNPY* attr);
   public:
        void render();


   public:
        const char*   getShaderDir();
        const char*   getShaderInclPath();
        Renderer*     getGeometryRenderer();
        Renderer*     getInstanceRenderer(unsigned int i);
        unsigned int  getNumInstanceRenderer();
   public:
        Rdr*          getAxisRenderer();
        Rdr*          getGenstepRenderer();
        Rdr*          getPhotonRenderer();
        Rdr*          getRecordRenderer();
        Rdr*          getRecordRenderer(RecordStyle_t style);
        //GMergedMesh*  getGeometry();
        Composition*  getComposition();
        NumpyEvt*     getNumpyEvt();
        Photons*      getPhotons();
        bool*         getModeAddress(const char* name);
        const char*   getRecordTag();
        float         getTimeFraction();

   private:
        char*        m_shader_dir ; 
        char*        m_shader_dynamic_dir ; 
        char*        m_shader_incl_path ; 
        Device*      m_device ; 
        Colors*      m_colors ; 
   private:
        unsigned int m_num_instance_renderer ; 
        Renderer*    m_geometry_renderer ; 
        Renderer*    m_instance_renderer[MAX_INSTANCE_RENDERER] ; 
        Renderer*    m_bbox_renderer[MAX_INSTANCE_RENDERER] ; 
        Renderer*    m_global_renderer ; 
        Renderer*    m_globalvec_renderer ; 
   private:
        Rdr*         m_axis_renderer ; 
        Rdr*         m_genstep_renderer ; 
        Rdr*         m_photon_renderer ; 
        Rdr*         m_record_renderer ; 
        Rdr*         m_altrecord_renderer ; 
        Rdr*         m_devrecord_renderer ; 
   private:
        NumpyEvt*    m_evt ;
        Photons*     m_photons ; 
        GGeo*        m_ggeo ;
        GMergedMesh* m_mesh0 ; 
        Composition* m_composition ;
        GBuffer*     m_colorbuffer ;
        unsigned int m_target ;
        unsigned int m_target_deferred ;
        unsigned int m_touch ;

   private:
        bool         m_global_mode ; 
        bool         m_globalvec_mode ; 
        bool         m_instance_mode[MAX_INSTANCE_RENDERER] ; 
        bool         m_bbox_mode[MAX_INSTANCE_RENDERER] ; 
        bool         m_axis_mode ; 
        bool         m_genstep_mode ; 
        bool         m_photon_mode ; 
        bool         m_record_mode ; 
   private:
        RecordStyle_t   m_record_style ; 
        GeometryStyle_t m_geometry_style ; 
        unsigned int    m_num_geometry_style ; 
        GlobalStyle_t   m_global_style ; 
        unsigned int    m_num_global_style ; 
        InstanceStyle_t m_instance_style ; 
        RenderStyle_t   m_render_style ; 
        bool            m_initialized ;  
        float           m_time_fraction ;  

     


};


inline Scene::Scene(const char* shader_dir, const char* shader_incl_path, const char* shader_dynamic_dir) 
            :
            m_shader_dir(shader_dir ? strdup(shader_dir): NULL ),
            m_shader_dynamic_dir(shader_dynamic_dir ? strdup(shader_dynamic_dir): NULL),
            m_shader_incl_path(shader_incl_path ? strdup(shader_incl_path): NULL),
            m_device(NULL),
            m_colors(NULL),
            m_num_instance_renderer(0),
            m_geometry_renderer(NULL),
            m_global_renderer(NULL),
            m_globalvec_renderer(NULL),
            m_axis_renderer(NULL),
            m_genstep_renderer(NULL),
            m_photon_renderer(NULL),
            m_record_renderer(NULL),
            m_altrecord_renderer(NULL),
            m_devrecord_renderer(NULL),
            m_evt(NULL),
            m_photons(NULL),
            m_ggeo(NULL),
            m_mesh0(NULL),
            m_composition(NULL),
            m_colorbuffer(NULL),
            m_target(0),
            m_target_deferred(0),
            m_touch(0),
            m_global_mode(false),
            m_globalvec_mode(false),
            m_axis_mode(true),
            m_genstep_mode(true),
            m_photon_mode(true),
            m_record_mode(true),
            m_record_style(REC),
            m_geometry_style(BBOX),
            m_num_geometry_style(0),
            m_global_style(GVIS),
            m_num_global_style(0),
            m_instance_style(IVIS),
            m_render_style(R_PROJECTIVE),
            m_initialized(false),
            m_time_fraction(0.f)
{

    init();

    for(unsigned int i=0 ; i < MAX_INSTANCE_RENDERER ; i++ ) 
    {
        m_instance_renderer[i] = NULL ; 
        m_bbox_renderer[i] = NULL ; 
        m_instance_mode[i] = false ; 
        m_bbox_mode[i] = false ; 
    }
}



inline const char* Scene::getShaderDir()
{
    return m_shader_dir ;
}
inline const char* Scene::getShaderInclPath()
{
    return m_shader_incl_path ;
}


inline void Scene::setGeometry(GGeo* gg)
{
    m_ggeo = gg ;
}
inline GGeo* Scene::getGeometry()
{
    return m_ggeo ; 
}

inline unsigned int Scene::getNumInstanceRenderer()
{
    return m_num_instance_renderer ; 
}

inline float Scene::getTimeFraction()
{
    return m_time_fraction ; 
}

inline unsigned int Scene::getTarget()
{
    return m_target ;
}
inline unsigned int Scene::getTouch()
{
    return m_touch ;
}
inline void Scene::setTouch(unsigned int touch)
{
    m_touch = touch ; 
}



inline Renderer* Scene::getGeometryRenderer()
{
    return m_geometry_renderer ; 
}


inline Rdr* Scene::getAxisRenderer()
{
    return m_axis_renderer ; 
}
inline Rdr* Scene::getGenstepRenderer()
{
    return m_genstep_renderer ; 
}
inline Rdr* Scene::getPhotonRenderer()
{
    return m_photon_renderer ; 
}



inline NumpyEvt* Scene::getNumpyEvt()
{
    return m_evt ; 
}
inline Photons* Scene::getPhotons()
{
    return m_photons ; 
}




//inline GMergedMesh* Scene::getGeometry()
//{
//    return m_geometry ; 
//}


inline void Scene::setNumpyEvt(NumpyEvt* evt)
{
    m_evt = evt ; 
}
inline void Scene::setPhotons(Photons* photons)
{
    m_photons = photons ; 
}




inline void Scene::setRecordStyle(RecordStyle_t style)
{
    m_record_style = style ; 
}

inline Scene::RecordStyle_t Scene::getRecordStyle()
{
    return m_record_style ; 
}







inline void Scene::nextPhotonStyle()
{
    int next = (m_record_style + 1) % NUM_RECORD_STYLE ; 
    m_record_style = (RecordStyle_t)next ; 
}




inline unsigned int Scene::getNumGeometryStyle()
{
    return m_num_geometry_style == 0 ? NUM_GEOMETRY_STYLE : m_num_geometry_style ;
}
inline void Scene::setNumGeometryStyle(unsigned int num_geometry_style)
{
    m_num_geometry_style = num_geometry_style ;
}



inline unsigned int Scene::getNumGlobalStyle()
{
    return m_num_global_style == 0 ? NUM_GLOBAL_STYLE : m_num_global_style ;
}
inline void Scene::setNumGlobalStyle(unsigned int num_global_style)
{
    m_num_global_style = num_global_style ;
}






inline void Scene::nextGeometryStyle()
{
    int next = (m_geometry_style + 1) % getNumGeometryStyle(); 
    setGeometryStyle( (GeometryStyle_t)next );

    const char* stylename = getGeometryStyleName();
    printf("Scene::nextGeometryStyle : %s \n", stylename);
}

inline void Scene::setGeometryStyle(GeometryStyle_t style)
{
    m_geometry_style = style ; 
    applyGeometryStyle();
}

inline void Scene::nextGlobalStyle()
{
    int next = (m_global_style + 1) % getNumGlobalStyle() ; 
    m_global_style = (GlobalStyle_t)next ; 
    applyGlobalStyle();
}



inline void Scene::applyGlobalStyle()
{
   // { GVIS, 
   //   GINVIS, 
   //   GVISVEC, 
   //   GVEC, 
   //   NUM_GLOBAL_STYLE }


    switch(m_global_style)
    {
        case GVIS:
                  m_global_mode = true ;    
                  m_globalvec_mode = false ;    
                  break ; 
        case GVISVEC:
                  m_global_mode = true ;    
                  m_globalvec_mode = true ;
                  break ; 
        case GVEC:
                  m_global_mode = false ;    
                  m_globalvec_mode = true ;
                  break ; 
        case GINVIS:
                  m_global_mode = false ;    
                  m_globalvec_mode = false ;
                  break ; 
        default:
                  assert(0);
        
    }
}







inline void Scene::nextRenderStyle()  // O:key
{
    int next = (m_render_style + 1) % NUM_RENDER_STYLE ; 
    m_render_style = (RenderStyle_t)next ; 
    applyRenderStyle();
}

inline void Scene::applyRenderStyle()   
{
    // nothing to do, style is honoured by  Scene::render
}
inline bool Scene::isProjectiveRender()
{
   return m_render_style == R_PROJECTIVE ;
}
inline bool Scene::isRaytracedRender()
{
   return m_render_style == R_RAYTRACED ;
}
inline bool Scene::isCompositeRender()
{
   return m_render_style == R_COMPOSITE ;
}


 


inline void Scene::nextInstanceStyle()
{
    int next = (m_instance_style + 1) % NUM_INSTANCE_STYLE ; 
    m_instance_style = (InstanceStyle_t)next ; 
    applyInstanceStyle();
}

inline void Scene::applyInstanceStyle()  // I:key 
{
    // hmm some overlap with GeometryStyle ... but that includes wireframe which can be very slow
    bool inst(false);
    switch(m_instance_style)
    {
        case IVIS:
                  inst = true ;    
                  break ; 
        case IINVIS:
                  inst = false ;    
                  break ; 
         default:
                  assert(0);
        
    }

   for(unsigned int i=0 ; i < m_num_instance_renderer ; i++ ) 
   {
       m_instance_mode[i] = inst ; 
       //m_bbox_mode[i] = !inst ; 
   } 

}
















inline unsigned int Scene::getTargetDeferred()
{
    return m_target_deferred ; 
}


