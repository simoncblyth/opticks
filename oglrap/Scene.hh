/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#pragma once

#include <glm/fwd.hpp>
#include "plog/Severity.h"

// brap-
class BDynamicDefine ; 

template <typename T> class NPY ; 

class Opticks ; 
class OpticksHub ; 
class OpticksEvent ; 
class MultiViewNPY ;

// ggeo-
class GDrawable ;
class GMergedMesh ;
class GGeoLib ;
class GBoundaryLibMetadata ;

// oglrap-

struct RContext ; 

class Renderer ; 
class InstLODCull ; 
class Rdr ;
class Device ; 
class Composition ; 
class ContentStyle ; 
class Photons ; 
class Colors ; 
class Interactor ; 

/*
Scene
======

Manages OpenGL rendering of geometry and event data.

* In principal could have multiple flavors of "Scene" targetting different Graphics
  APIs (eg Metal, Vulkan, DirectX, OpenGL ES, SceneKit ) 

* Thus things that are not dependant on OpenGL should be done elsewhere, 
  eg LODification of meshes would need to be done for all Scene flavors,
  so it does not belong here, up in OpticksViz is more appropriate 


Canonical m_scene instance is a resident of OpticksViz

=======================  ===========      ===========   ==================================================================================
Constituent               Class            Shader Tag    Notes
=======================  ===========      ===========   ==================================================================================
m_global_renderer         Renderer         nrm
m_globalvec_renderer      Renderer         nrmvec        global renderer with face normals represented by geometry shader generated lines
m_raytrace_renderer       Renderer         tex           great big quad rendering of raytrace texture obtained from OptiX
m_instance_renderer[i]    Renderer         inrm          
m_instlodcull[i]          InstLODCull      inrmcull      WIP
m_bbox_renderer[i]        Renderer         inrm   
m_axis_renderer           Rdr              axis
m_genstep_renderer        Rdr              p2l
m_nopstep_renderer        Rdr              nop           LINE_STRIP primitive
m_photon_renderer         Rdr              pos
m_source_renderer         Rdr              pos
m_record_renderer         Rdr              rec           LINE_STRIP primitive
m_altrecord_renderer      Rdr              altrec        LINE_STRIP primitive
m_devrecord_renderer      Rdr              devrec        LINE_STRIP primitive
=======================  ===========      ===========   ==================================================================================


uploadGeometryGlobal
    uploads to m_global_renderer  and m_globalvec_renderer

uploadGeometryInstanced
    uploads to m_instance_renderer[i] and m_bbox_renderer[i] the bbox mesh being generated from the GMergedMesh 

uploadAxis
    uploads to m_axis_renderer

uploadEvent
    uploads to m_genstep_renderer, m_nopstep_renderer, m_photon_renderer, m_record_renderer, m_altrecord_renderer, m_devrecord_renderer

uploadEventSelection
    uploads to m_photon_renderer, m_record_renderer, m_altrecord_renderer, m_devrecord_renderer


Despite appearances buffers are not uploaded more than once, however binding must be done for every renderer.



ContentStyle (B key)
    what combination of bbox and instances to render

GlobalStyle (Q key)
    control global and globalvec rendering of non-instanced geometry

InstanceStyle (I key)
    control instance rendering (some overlap with ContentStyle) 



*/


#include "NConfigurable.hpp"
#include "SCtrl.hh"
#include "OGLRAP_API_EXPORT.hh"

class OGLRAP_API Scene : public NConfigurable, public SCtrl  {
        static const plog::Severity LEVEL ; 
   public:
        static Scene*      GetInstance();
        static const char* PREFIX ;
        const char* getPrefix();
   public:
        static const char* PHOTON ;
        static const char* SOURCE ;
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
        enum { MAX_INSTANCE_RENDERER = 15 };  
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
        // SCtrl
        void command(const char* cmd); 
   public:
        void setVerbosity(unsigned verbosity);
        void setRenderMode(const char* s);
        std::string getRenderMode() const ;
        void dump_uploads_table(const char* msg="Scene::dump_uploads_table");
        std::string desc() const ;
   public:
        // P-key
        typedef enum { REC, ALTREC, DEVREC, NUM_RECORD_STYLE } RecordStyle_t ;
        void setRecordStyle(Scene::RecordStyle_t style);
        Scene::RecordStyle_t getRecordStyle();
        static const char* getRecordStyleName(Scene::RecordStyle_t style);
        const char* getRecordStyleName();
        //void nextPhotonStyle();
        void nextRecordStyle();
        void commandRecordStyle(const char* cmd);
   public:
        // B-key : mostly moved to okc.ContentStyle
        void applyContentStyle(); 
   public:
        void setWireframe(bool wire=true);
        void setInstCull(bool instcull=true);
   public:
        // MINUS-key : skip geometry rendering 
        typedef enum { NOSKIPGEO, SKIPGEO, NUM_SKIPGEO_STYLE } SkipGeoStyle_t ;  
        void nextSkipGeoStyle();
        void commandSkipGeoStyle(const char* cmd);
        void setSkipGeoStyle(int style);
   public:
        // EQUAL-key : skip event rendering 
        typedef enum { NOSKIPEVT, SKIPEVT, NUM_SKIPEVT_STYLE } SkipEvtStyle_t ;  
        void nextSkipEvtStyle();
        void commandSkipEvtStyle(const char* cmd);
        void setSkipEvtStyle(int style);
   public:
        // I-key
        typedef enum { IVIS, IINVIS, NUM_INSTANCE_STYLE } InstanceStyle_t ;  
        void nextInstanceStyle();
        void commandInstanceStyle(const char* cmd);
        void setInstanceStyle(int style);
        void applyInstanceStyle();
   public:
        Scene(OpticksHub* hub);
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
        //void setComposition(Composition* composition);
        void hookupRenderers();
        void setPhotons(Photons* photons);
   public:
        void setGeometry(GGeoLib* geolib);
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
      //   migrated targetting to OpticksHub/OpticksGeometry   
      //
      //  void setTarget(unsigned int index=0, bool aim=true); 
      //  unsigned int getTargetDeferred();


        unsigned int getTarget(); 
        void setTarget(unsigned int index=0, bool aim=true); 

        int touch(int ix, int iy, float depth);
        void setTouch(unsigned int index); 
        unsigned int getTouch();
        void jump(); 

   public:
        void upload(OpticksEvent* evt);
   private:
        void uploadEvent(OpticksEvent* evt);
        void uploadEventSelection(OpticksEvent* evt);
        void uploadAxis();
   private:
        void uploadRecordAttr(MultiViewNPY* attr, bool debug=false);
   public:
        void render();
        void preRenderCompute();
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
        Rdr*          getSourceRenderer();
        Rdr*          getRecordRenderer();
        Rdr*          getRecordRenderer(RecordStyle_t style);
        Composition*  getComposition();
        Photons*      getPhotons();
        bool*         getModeAddress(const char* name);
        const char*   getRecordTag();
        float         getTimeFraction();

   private:
        static Scene* fInstance ; 
        OpticksHub*  m_hub ; 
        Opticks*     m_ok ;     // think twice before using this, is the feature OpenGL specific ? Does it belong here ?
        char*        m_shader_dir ; 
        char*        m_shader_dynamic_dir ; 
        char*        m_shader_incl_path ; 
        Device*      m_device ; 
        Colors*      m_colors ; 
        RContext*    m_context ; 

        Interactor*  m_interactor ; 
   private:
        unsigned int m_num_instance_renderer ; 
        Renderer*    m_geometry_renderer ; 
        InstLODCull* m_instlodcull[MAX_INSTANCE_RENDERER] ; 
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
        Rdr*         m_source_renderer ; 
        Rdr*         m_record_renderer ; 
        Rdr*         m_altrecord_renderer ; 
        Rdr*         m_devrecord_renderer ; 
   private:
        Photons*     m_photons ; 
        GGeoLib*     m_geolib ;
        GMergedMesh* m_mesh0 ; 
        Composition* m_composition ;
        ContentStyle*   m_content_style ; 
        NPY<unsigned char>*     m_colorbuffer ;
        unsigned int m_touch ;

   private:
        bool*        m_global_mode_ptr ; 
        bool*        m_globalvec_mode_ptr ; 

        bool         m_instance_mode[MAX_INSTANCE_RENDERER] ; 
        bool         m_bbox_mode[MAX_INSTANCE_RENDERER] ; 
        bool         m_axis_mode ; 
        bool         m_genstep_mode ; 
        bool         m_nopstep_mode ; 
        bool         m_photon_mode ; 
        bool         m_source_mode ; 
        bool         m_record_mode ; 
   private:
        RecordStyle_t   m_record_style ; 
        InstanceStyle_t m_instance_style ; 
        SkipGeoStyle_t  m_skipgeo_style ; 
        SkipEvtStyle_t  m_skipevt_style ; 

   private:
        bool            m_initialized ;  
        float           m_time_fraction ;  
        bool            m_instcull ; 
        unsigned        m_verbosity ; 
        unsigned        m_render_count ; 

};



