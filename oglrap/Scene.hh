#pragma once

#include <glm/fwd.hpp>

// brap-
class BDynamicDefine ; 

template <typename T> class NPY ; 

class OpticksEvent ; 
class MultiViewNPY ;

// ggeo-
class GDrawable ;
class GMergedMesh ;
class GGeo ;
class GBoundaryLibMetadata ;

// oglrap-
class Renderer ; 
class Rdr ;
class Device ; 
class Composition ; 
class Photons ; 
class Colors ; 
class Interactor ; 


#include "NConfigurable.hpp"
#include "OGLRAP_API_EXPORT.hh"
class OGLRAP_API Scene : public NConfigurable {
   public:
        static const char* PREFIX ;
        const char* getPrefix();
   public:
        static const char* PHOTON ;
        static const char* AXIS ;
        static const char* GENSTEP ;
        static const char* NOPSTEP ;
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
        void dump_uploads_table(const char* msg="Scene::dump_uploads_table");
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
        void nextRenderStyle(unsigned int modifiers);  // key O (formerly optix render mode toggle)
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
        void write(BDynamicDefine* dd);
        void gui();
        void initRenderers();
        void initRenderersDebug(); // debugging interop buffer overwrite issue with subset of renderers
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
        void setEvent(OpticksEvent* evt);
        void setPhotons(Photons* photons);
   public:
        void setGeometry(GGeo* gg);
        GGeo* getGeometry();
        void uploadGeometry(); 
   private:
        void uploadGeometryGlobal(GMergedMesh* mm);
        void uploadGeometryInstanced(GMergedMesh* mm);
   public:
        void uploadColorBuffer(NPY<unsigned char>* colorbuffer);
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
        void uploadRecordAttr(MultiViewNPY* attr, bool debug=false);
   public:
        void render();
        void renderGeometry();
        void renderEvent();

   public:
        void setInteractor(Interactor* interactor);
   public:
        const char*   getShaderDir();
        const char*   getShaderInclPath();
        Interactor*   getInteractor();
        Renderer*     getGeometryRenderer();
        Renderer*     getRaytraceRenderer();
        Renderer*     getInstanceRenderer(unsigned int i);
        unsigned int  getNumInstanceRenderer();
   public:
        Rdr*          getAxisRenderer();
        Rdr*          getGenstepRenderer();
        Rdr*          getNopstepRenderer();
        Rdr*          getPhotonRenderer();
        Rdr*          getRecordRenderer();
        Rdr*          getRecordRenderer(RecordStyle_t style);
        //GMergedMesh*  getGeometry();
        Composition*  getComposition();
        OpticksEvent* getEvent();
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
        Interactor*  m_interactor ; 
   private:
        unsigned int m_num_instance_renderer ; 
        Renderer*    m_geometry_renderer ; 
        Renderer*    m_instance_renderer[MAX_INSTANCE_RENDERER] ; 
        Renderer*    m_bbox_renderer[MAX_INSTANCE_RENDERER] ; 
        Renderer*    m_global_renderer ; 
        Renderer*    m_globalvec_renderer ; 
        Renderer*    m_raytrace_renderer ; 
   private:
        Rdr*         m_axis_renderer ; 
        Rdr*         m_genstep_renderer ; 
        Rdr*         m_nopstep_renderer ; 
        Rdr*         m_photon_renderer ; 
        Rdr*         m_record_renderer ; 
        Rdr*         m_altrecord_renderer ; 
        Rdr*         m_devrecord_renderer ; 
   private:
        OpticksEvent*    m_evt ;
        Photons*     m_photons ; 
        GGeo*        m_ggeo ;
        GMergedMesh* m_mesh0 ; 
        Composition* m_composition ;
        NPY<unsigned char>*     m_colorbuffer ;
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
        bool         m_nopstep_mode ; 
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



