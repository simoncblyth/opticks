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

#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cstring>
#include <string>
#include <sstream>
#include <vector>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include "Scene.hh"
#include <GL/glew.h>


// brap-
#include "BDynamicDefine.hh"

// npy-
#include "NGLM.hpp"
#include "GLMPrint.hpp"
#include "GLMFormat.hpp"
#include "NPY.hpp"
#include "ViewNPY.hpp"
#include "MultiViewNPY.hpp"

// opticks-
#include "Opticks.hh"
#include "OpticksResource.hh"
#include "OpticksConst.hh"
#include "OpticksEvent.hh"

// okg-
#include "OpticksHub.hh"

// ggeo-
#include "GMergedMesh.hh"
#include "GBBoxMesh.hh"
#include "GGeoLib.hh"


// oglrap-
#include "OGLRap_Config.hh"      // cmake generated header

#include "Composition.hh"
#include "ContentStyle.hh"
#include "Renderer.hh"
#include "RContext.hh"
#include "InstLODCull.hh"
#include "Device.hh"
#include "Rdr.hh"
#include "Colors.hh"
#include "Interactor.hh"

#include "OGLRap_imgui.hh"

#include "PLOG.hh"

const plog::Severity Scene::LEVEL = PLOG::EnvLevel("Scene", "DEBUG"); 

Scene* Scene::fInstance = NULL ; 
Scene* Scene::GetInstance(){ return fInstance ; }


const char* Scene::PREFIX = "scene" ;
const char* Scene::getPrefix()
{
   return PREFIX ; 
}




const char* Scene::AXIS   = "axis" ; 
const char* Scene::PHOTON = "photon" ; 
const char* Scene::SOURCE = "source" ; 
const char* Scene::GENSTEP = "genstep" ; 
const char* Scene::NOPSTEP = "nopstep" ; 
const char* Scene::GLOBAL  = "global" ; 

const char* Scene::_INSTANCE  = "in" ; 
const char* Scene::INSTANCE0 = "in0" ; 
const char* Scene::INSTANCE1 = "in1" ; 
const char* Scene::INSTANCE2 = "in2" ; 
const char* Scene::INSTANCE3 = "in3" ; 
const char* Scene::INSTANCE4 = "in4" ; 

const char* Scene::_BBOX      = "bb" ; 
const char* Scene::BBOX0     = "bb0" ; 
const char* Scene::BBOX1     = "bb1" ; 
const char* Scene::BBOX2     = "bb2" ; 
const char* Scene::BBOX3     = "bb3" ; 
const char* Scene::BBOX4     = "bb4" ; 

const char* Scene::RECORD   = "record" ; 

const char* Scene::REC_ = "point" ;   
const char* Scene::ALTREC_ = "line" ; 
const char* Scene::DEVREC_ = "vector" ; 


const char* Scene::getRecordStyleName(Scene::RecordStyle_t style)
{
   switch(style)
   {
      case    REC:return REC_ ; break; 
      case ALTREC:return ALTREC_ ; break; 
      case DEVREC:return DEVREC_ ; break; 
      case NUM_RECORD_STYLE:assert(0) ; break ; 
      default: assert(0); break ; 
   } 
   return NULL ; 
}

const char* Scene::getRecordStyleName()
{
   return getRecordStyleName(getRecordStyle());
}

/**
Scene::Scene
--------------

Instanciated from OpticksViz::init 


**/

Scene::Scene(OpticksHub* hub) 
    :
    m_hub(hub),
    m_ok(hub->getOpticks()),
    m_shader_dir(NULL),
    m_shader_dynamic_dir(NULL),
    m_shader_incl_path(NULL),
    m_device(NULL),
    m_colors(NULL),
    m_interactor(NULL),
    m_num_instance_renderer(0),
    m_geometry_renderer(NULL),
    m_global_renderer(NULL),
    m_globalvec_renderer(NULL),
    m_raytrace_renderer(NULL),
    m_axis_renderer(NULL),
    m_genstep_renderer(NULL),
    m_nopstep_renderer(NULL),
    m_photon_renderer(NULL),
    m_source_renderer(NULL),
    m_record_renderer(NULL),
    m_altrecord_renderer(NULL),
    m_devrecord_renderer(NULL),
    m_photons(NULL),
    m_geolib(NULL),
    m_mesh0(NULL),
    m_composition(m_hub->getComposition()),
    m_colorbuffer(NULL),
    m_touch(0),
    m_global_mode_ptr(NULL),
    m_globalvec_mode_ptr(NULL),
    m_axis_mode(true),
    m_genstep_mode(true),
    m_nopstep_mode(true),
    m_photon_mode(true),
    m_source_mode(false),
    m_record_mode(true),
    m_record_style(ALTREC),
    m_instance_style(IVIS),
    m_skipgeo_style(NOSKIPGEO),
    m_skipevt_style(NOSKIPEVT),
    m_initialized(false),
    m_time_fraction(0.f),
    m_instcull(true),
    m_verbosity(0),
    m_render_count(0)
{
    init();
    fInstance = this ; 

    for(unsigned int i=0 ; i < MAX_INSTANCE_RENDERER ; i++ ) 
    {
        m_instance_renderer[i] = NULL ; 
        m_instlodcull[i] = NULL ; 
        m_bbox_renderer[i] = NULL ; 
        m_instance_mode[i] = false ; 
        m_bbox_mode[i] = false ; 
    }
}

 
void Scene::init()
{
    m_content_style = m_composition->getContentStyle(); 
    m_global_mode_ptr = m_composition->getGlobalModePtr(); 
    m_globalvec_mode_ptr = m_composition->getGlobalVecModePtr(); 


/*
    LOG(LEVEL)
          << " OGLRAP_INSTALL_PREFIX " << OGLRAP_INSTALL_PREFIX
          << " OGLRAP_SHADER_DIR " << OGLRAP_SHADER_DIR
          << " OGLRAP_SHADER_INCL_PATH " << OGLRAP_SHADER_INCL_PATH
          << " OGLRAP_SHADER_DYNAMIC_DIR " << OGLRAP_SHADER_DYNAMIC_DIR
          ;   
*/

    const char* shader_dir = OpticksResource::ShaderDir(); 
    m_shader_dir = strdup(shader_dir) ; 
    m_shader_incl_path = strdup(shader_dir);
    m_shader_dynamic_dir = strdup(shader_dir);
}

void Scene::write(BDynamicDefine* dd)
{
    LOG(LEVEL) << "shader_dynamic_dir " << m_shader_dynamic_dir ; 
    dd->write( m_shader_dynamic_dir, "dynamic.h" );
}

void Scene::setRenderMode(const char* s)
{
    // setting renderer toggles

    LOG(LEVEL)
        << " [" << s  << "] "
        ;

    std::vector<std::string> elem ; 
    boost::split(elem, s, boost::is_any_of(","));
    
    for(unsigned int i=0 ; i < elem.size() ; i++)
    {
        const char* elem_ = elem[i].c_str();
        const char* el ; 
        bool setting = true ; 
        if(elem_[0] == '-' || elem_[0] == '+')
        {
            setting = elem_[0] == '-' ? false : true ;
            el = elem_ + 1 ; 
        }
        else
        {
            el = elem_ ; 
        }

        if(strncmp(el, _BBOX, strlen(_BBOX))==0) 
        {
             unsigned int bbx = boost::lexical_cast<unsigned int>(el+strlen(_BBOX)) ;
             if(bbx < MAX_INSTANCE_RENDERER)
                  *(m_bbox_mode+bbx) = setting ;  
        } 
        if(strncmp(el, _INSTANCE, strlen(_INSTANCE))==0) 
        {
             const char* idx = el+strlen(_INSTANCE) ;
             if(strlen(idx) == 0)  // +in
             {
                  for(unsigned ins=0 ; ins < MAX_INSTANCE_RENDERER ; ins++) 
                     *(m_instance_mode+ins) = setting ;  
             }
             else         // +in0 +in1 ...
             {
                 unsigned int ins = boost::lexical_cast<unsigned int>(idx) ;
                 if(ins < MAX_INSTANCE_RENDERER)
                      *(m_instance_mode+ins) = setting ;  
             }
        } 
        
        if(strcmp(el, GLOBAL)==0)  *m_global_mode_ptr = setting ; 
        if(strcmp(el, AXIS)==0)    m_axis_mode = setting ; 
        if(strcmp(el, GENSTEP)==0) m_genstep_mode = setting ; 
        if(strcmp(el, NOPSTEP)==0) m_nopstep_mode = setting ; 
        if(strcmp(el, PHOTON)==0)  m_photon_mode = setting ; 
        if(strcmp(el, SOURCE)==0)  m_source_mode = setting ; 
        if(strcmp(el, RECORD)==0)  m_record_mode = setting ; 
    }
}

std::string Scene::getRenderMode() const 
{
    const char* delim = "," ; 

    std::stringstream ss ; 

    if(*m_global_mode_ptr)  ss << GLOBAL << delim ; 
    if(m_axis_mode)    ss << AXIS << delim ; 
    if(m_genstep_mode) ss << GENSTEP << delim ; 
    if(m_nopstep_mode) ss << NOPSTEP << delim ; 
    if(m_photon_mode) ss << PHOTON << delim ; 
    if(m_source_mode) ss << SOURCE << delim ; 
    if(m_record_mode) ss << RECORD << delim ; 

    for(unsigned int i=0 ; i<MAX_INSTANCE_RENDERER ; i++) if(m_instance_mode[i]) ss << _INSTANCE << i << delim ; 
    for(unsigned int i=0 ; i<MAX_INSTANCE_RENDERER ; i++) if(m_bbox_mode[i]) ss << _BBOX << i << delim ; 

    return ss.str();
}





void Scene::gui()
{
#ifdef GUI_
     ImGui::Checkbox(GLOBAL,    m_composition->getGlobalModePtr() );

     ImGui::Checkbox(BBOX0,     m_bbox_mode+0);
     ImGui::Checkbox(BBOX1,     m_bbox_mode+1);
     ImGui::Checkbox(BBOX2,     m_bbox_mode+2);
     ImGui::Checkbox(BBOX3,     m_bbox_mode+3);
     ImGui::Checkbox(BBOX4,     m_bbox_mode+4);

     ImGui::Checkbox(INSTANCE0, m_instance_mode+0);
     ImGui::Checkbox(INSTANCE1, m_instance_mode+1);
     ImGui::Checkbox(INSTANCE2, m_instance_mode+2);
     ImGui::Checkbox(INSTANCE3, m_instance_mode+3);
     ImGui::Checkbox(INSTANCE4, m_instance_mode+4);

     ImGui::Checkbox(AXIS,     &m_axis_mode);
     ImGui::Checkbox(GENSTEP,  &m_genstep_mode);
     ImGui::Checkbox(NOPSTEP,  &m_nopstep_mode);
     ImGui::Checkbox(PHOTON,   &m_photon_mode);
     ImGui::Checkbox(SOURCE,   &m_source_mode);
     ImGui::Checkbox(RECORD,   &m_record_mode);
    // ImGui::Text(" target: %u ", m_target );
     ImGui::Text(" genstep %d nopstep %d photon %d source %d record %d \n", 
             ( m_genstep_renderer ? m_genstep_renderer->getCountDefault() : -1 ),
             ( m_nopstep_renderer ? m_nopstep_renderer->getCountDefault() : -1 ),
             ( m_photon_renderer ? m_photon_renderer->getCountDefault() : -1 ),
             ( m_source_renderer ? m_source_renderer->getCountDefault() : -1 ),
             ( m_record_renderer ? m_record_renderer->getCountDefault() : -1 )
     );



     int* record_style = (int*)&m_record_style ;       // address of enum cast to int*
     ImGui::RadioButton("rec",    record_style, REC); 
     ImGui::SameLine();
     ImGui::RadioButton("altrec", record_style, ALTREC); 
     ImGui::SameLine();
     ImGui::RadioButton("devrec", record_style, DEVREC); 


#endif    
}

const char* Scene::TARGET = "scenetarget" ; // trying to extracate targetting from Scene 

bool Scene::accepts(const char* name)
{
    return 
          strcmp(name, TARGET) == 0  ;
}  

std::vector<std::string> Scene::getTags()
{
    std::vector<std::string> tags ;
    tags.push_back(TARGET);
    return tags ; 
}


std::string Scene::get(const char* name)
{
    int v(0) ; 
    if(     strcmp(name,TARGET)==0) v = getTarget();
    else
         printf("Scene::get bad name %s\n", name);

    return gformat(v);
}

void Scene::set(const char* name, std::string& s)
{
    int v = gint_(s); 
    if(     strcmp(name,TARGET)==0)    setTarget(v);
    else
         printf("Scene::set bad name %s\n", name);
}

void Scene::configure(const char* name, const char* value_)
{
    std::string val(value_);
    int value = gint_(val); 
    configure(name, value);
}

void Scene::configureI(const char* name, std::vector<int> values)
{
    LOG(info) << "Scene::configureI";
    if(values.empty()) return ;
    int last = values.back();
    configure(name, last);
}

void Scene::configure(const char* name, int value)
{
    if(strcmp(name, TARGET) == 0)
    {
        setTarget(value);   
    }
    else
    {
        LOG(warning)<<"Scene::configure ignoring " << name << " " << value ;
    }
}


void Scene::setInstCull(bool instcull)
{
    m_instcull = instcull ; 
}

void Scene::setWireframe(bool wire)
{
    if(m_global_renderer)
        m_global_renderer->setWireframe(wire);

    for( unsigned int i=0 ; i < MAX_INSTANCE_RENDERER ; i++)
    {
        if(m_instance_renderer[i])
            m_instance_renderer[i]->setWireframe(wire);

        if(m_bbox_renderer[i])
            m_bbox_renderer[i]->setWireframe(false);  

        // wireframe is much slower than filled, 
        // also bbox winding order is not correct
        // so keeping the bbox as filled
    }
}



void Scene::initRenderersDebug()
{
    m_device = new Device();

    m_colors = new Colors(m_device);

    m_genstep_renderer = new Rdr(m_device, "p2l", m_shader_dir, m_shader_incl_path);

    m_photon_renderer = new Rdr(m_device, "pos", m_shader_dir, m_shader_incl_path );

    m_source_renderer = new Rdr(m_device, "pos", m_shader_dir, m_shader_incl_path );


    for( unsigned int i=0 ; i < MAX_INSTANCE_RENDERER ; i++)
    {
        m_instance_mode[i] = false ; 
        m_instance_renderer[i] = NULL ; 
        m_instlodcull[i] = NULL ; 

        m_bbox_mode[i] = false ; 
        m_bbox_renderer[i] = NULL ;
    }

    m_initialized = true ; 
}


void Scene::initRenderers()
{
    LOG(debug) << "Scene::initRenderers " 
              << " shader_dir " << m_shader_dir 
              << " shader_incl_path " << m_shader_incl_path 
               ;
   
    assert(m_shader_dir);

    m_device = new Device();

    m_colors = new Colors(m_device);

    m_context = new RContext ;  // UBO provisioning for InstLODCull

    m_global_renderer = new Renderer("nrm", m_shader_dir, m_shader_incl_path );
    m_globalvec_renderer = new Renderer("nrmvec", m_shader_dir, m_shader_incl_path );
    m_raytrace_renderer = new Renderer("tex", m_shader_dir, m_shader_incl_path );
    // m_raytrace_renderer just presents textures passed to it 
    // to follow how this gets fed, see opticksgl/OKGLTracer.cc::render


   // small array of instance renderers to handle multiple assemblies of repeats 
    for( unsigned int i=0 ; i < MAX_INSTANCE_RENDERER ; i++)
    {
        //m_instance_mode[i] = false ; 

        m_instance_renderer[i] = new Renderer("inrm", m_shader_dir, m_shader_incl_path );
        m_instance_renderer[i]->setInstanced();
        m_instance_renderer[i]->setIndexBBox(i, false);

        if(m_instcull)
        {
            m_instlodcull[i] = new InstLODCull("inrmcull", m_shader_dir, m_shader_incl_path);
            m_instlodcull[i]->setVerbosity(1);
            m_instlodcull[i]->setIndexBBox(i, false);
            m_instance_renderer[i]->setInstLODCull(m_instlodcull[i]);
        }

        //m_bbox_mode[i] = false ; 
        m_bbox_renderer[i] = new Renderer("inrm", m_shader_dir, m_shader_incl_path );
        m_bbox_renderer[i]->setInstanced();
        m_bbox_renderer[i]->setIndexBBox(i, true);
        m_bbox_renderer[i]->setWireframe(false);  // wireframe is much slower than filled
    }

    //LOG(info) << "Scene::init geometry_renderer ctor DONE";

    m_axis_renderer = new Rdr(m_device, "axis", m_shader_dir, m_shader_incl_path );

    m_genstep_renderer = new Rdr(m_device, "p2l", m_shader_dir, m_shader_incl_path);

    m_nopstep_renderer = new Rdr(m_device, "nop", m_shader_dir, m_shader_incl_path);
    m_nopstep_renderer->setPrimitive(Rdr::LINE_STRIP);

    m_photon_renderer = new Rdr(m_device, "pos", m_shader_dir, m_shader_incl_path );

    m_source_renderer = new Rdr(m_device, "pos", m_shader_dir, m_shader_incl_path );


    //
    // RECORD RENDERING USES AN UNPARTIONED BUFFER OF ALL RECORDS
    // SO THE GEOMETRY SHADERS HAVE TO THROW INVALID STEPS AS DETERMINED BY
    // COMPARING THE TIMES OF THE STEP PAIRS  
    // THIS MEANS SINGLE VALID STEPS WOULD BE IGNORED..
    // THUS MUST SUPPLY LINE_STRIP SO GEOMETRY SHADER CAN GET TO SEE EACH VALID
    // VERTEX IN A PAIR
    //
    // OTHERWISE WILL MISS STEPS
    //
    //  see explanations in gl/altrec/geom.glsl
    //
    m_record_renderer = new Rdr(m_device, "rec", m_shader_dir, m_shader_incl_path );
    m_record_renderer->setPrimitive(Rdr::LINE_STRIP);

    m_altrecord_renderer = new Rdr(m_device, "altrec", m_shader_dir, m_shader_incl_path);
    m_altrecord_renderer->setPrimitive(Rdr::LINE_STRIP);

    m_devrecord_renderer = new Rdr(m_device, "devrec", m_shader_dir, m_shader_incl_path);
    m_devrecord_renderer->setPrimitive(Rdr::LINE_STRIP);

    m_initialized = true ; 
}

void Scene::hookupRenderers()
{
    // HMM better to use hub ? to avoid this lot ?
    //     actually best to move to 1 or 2 UBO 
    //     updated from Scene, then most Renderers wont need composition ? 

    assert(m_composition); 
    Composition* composition = m_composition ; 

    if(m_global_renderer)
        m_global_renderer->setComposition(composition);

    if(m_globalvec_renderer)
        m_globalvec_renderer->setComposition(composition);

    if(m_raytrace_renderer)
        m_raytrace_renderer->setComposition(composition);

    // set for all instance slots, otherwise requires setComposition after uploadGeometry
    // as only then is m_num_instance_renderer set
    for( unsigned int i=0 ; i < MAX_INSTANCE_RENDERER ; i++)    
    {
        if(m_instance_renderer[i])
            m_instance_renderer[i]->setComposition(composition);

        if(m_bbox_renderer[i])
            m_bbox_renderer[i]->setComposition(composition);
    }

    if(m_axis_renderer)
        m_axis_renderer->setComposition(composition);

    if(m_genstep_renderer)
        m_genstep_renderer->setComposition(composition);

    if(m_nopstep_renderer)
         m_nopstep_renderer->setComposition(composition);

    if(m_photon_renderer)
        m_photon_renderer->setComposition(composition);

    if(m_source_renderer)
        m_source_renderer->setComposition(composition);


    if(m_record_renderer)
        m_record_renderer->setComposition(composition);

    if(m_altrecord_renderer)
        m_altrecord_renderer->setComposition(composition);

    if(m_devrecord_renderer)
        m_devrecord_renderer->setComposition(composition);
}



void Scene::uploadGeometryGlobal(GMergedMesh* mm)
{
    LOG(LEVEL)<< "[" ;

    assert(m_mesh0 == NULL); // not expected to Scene::uploadGeometryGlobal more than once 
    m_mesh0 = mm ; 

    bool skip = mm == NULL ? true : mm->isSkip() ;

    static unsigned int n_global(0);

    if(!skip)
    {
        if(m_global_renderer)
            m_global_renderer->upload(mm);  
        if(m_globalvec_renderer)
            m_globalvec_renderer->upload(mm);   // buffers are not re-uploaded, but binding must be done for each renderer 

        n_global++ ; 
        assert(n_global == 1);
    }
    else
    {
         LOG(error) << "SKIPPING GLOBAL " ; 
    }
    LOG(LEVEL)<< "]" ;
}


void Scene::uploadGeometryInstanced(GMergedMesh* mm)
{
    bool empty = mm->isEmpty();
    bool skip = mm->isSkip() ;

    if(!skip && !empty)
    { 
        assert(m_num_instance_renderer < MAX_INSTANCE_RENDERER) ;
        LOG(LEVEL)<< "instance renderer " << m_num_instance_renderer << " instcull " << m_instcull ;

        NPY<float>* ibuf = mm->getITransformsBuffer();
        assert(ibuf);

        if(m_instance_renderer[m_num_instance_renderer])
        {
            m_instance_renderer[m_num_instance_renderer]->setLOD(m_ok->getLOD());
            m_instance_renderer[m_num_instance_renderer]->upload(mm);
            //m_instance_mode[m_num_instance_renderer] = true ; 
        }

        LOG(verbose)<< "num_instance_renderer " << m_num_instance_renderer  ;
        GBBoxMesh* bb = GBBoxMesh::create(mm); assert(bb);

        if(m_bbox_renderer[m_num_instance_renderer])
        {
            m_bbox_renderer[m_num_instance_renderer]->upload(bb);
            //m_bbox_mode[m_num_instance_renderer] = true ; 
        }
        m_num_instance_renderer++ ; 
    }
    else
    {
         LOG(error) 
             << "SKIPPING " 
             << " empty " << empty 
             << " skip " << skip 
             ; 
    }
}


void Scene::uploadGeometry()
{
    // invoked by OpticksViz::uploadGeometry
    assert(m_geolib && "must setGeometry first");
    unsigned int nmm = m_geolib->getNumMergedMesh();

    LOG(info) << " nmm " << nmm ;

    m_geolib->dump("Scene::uploadGeometry GGeoLib" );

    m_context->init();  // UBO setup


    for(unsigned int i=0 ; i < nmm ; i++)
    {
        GMergedMesh* mm = m_geolib->getMergedMesh(i);
        if(!mm) continue ; 

        LOG(debug) << i << " geoCode " << mm->getGeoCode() ; 

        if( i == 0 )  // first mesh assumed to be **the one and only** non-instanced global mesh
        {
           uploadGeometryGlobal(mm);
        }
        else
        {
           uploadGeometryInstanced(mm);
        }
    }

    LOG(LEVEL)
        << " m_num_instance_renderer " << m_num_instance_renderer
        ; 

    applyContentStyle(); // sets m_instance_mode m_bbox_mode switches, change with "B"  nextContentStyle()
}

void Scene::uploadColorBuffer(NPY<unsigned char>* colorbuffer)
{
   // invoked by OpticksViz::uploadGeometry
    m_colorbuffer = colorbuffer ; 
    m_colors->setColorBuffer(colorbuffer);
    m_colors->upload();
}

Rdr* Scene::getRecordRenderer()
{
    return getRecordRenderer(m_record_style);
}

Rdr* Scene::getRecordRenderer(RecordStyle_t style)
{
    Rdr* rdr = NULL ; 
    switch(style)
    {
        case      REC:rdr = m_record_renderer     ; break ;
        case   ALTREC:rdr = m_altrecord_renderer  ; break ;
        case   DEVREC:rdr = m_devrecord_renderer  ; break ;
        case   NUM_RECORD_STYLE:                  ; break ;
    }
    return rdr ; 
}



void Scene::upload(OpticksEvent* evt)
{
    LOG(LEVEL) << "[" ;
        
    uploadAxis();

    LOG(debug) << "uploadAxis  DONE " ;

    uploadEvent(evt);  // Scene, Rdr uploads orchestrated by OpticksEvent/MultiViewNPY

    LOG(debug) << "uploadEvt  DONE " ;

    uploadEventSelection(evt);   // recsel upload

    LOG(debug) << "uploadSelection  DONE " ;

    LOG(LEVEL) << "]" ;
}



void Scene::uploadAxis()
{
    if(m_axis_renderer)
        m_axis_renderer->upload(m_composition->getAxisAttr());
}

void Scene::uploadEvent(OpticksEvent* evt)
{
    if(!evt) 
    {
       LOG(fatal) << "no evt " ;
       assert(evt);
    }

    // The Rdr call glBufferData using bytes and size from the associated NPY 
    // the bytes used is NULL when npy->hasData() == false
    // corresponding to device side only OpenGL allocation

    if(m_genstep_renderer)
        m_genstep_renderer->upload(evt->getGenstepAttr());

    if(m_nopstep_renderer) 
         m_nopstep_renderer->upload(evt->getNopstepAttr());

    if(m_photon_renderer)
         m_photon_renderer->upload(evt->getPhotonAttr());

    if(m_source_renderer)
         m_source_renderer->upload(evt->getSourceAttr());


    uploadRecordAttr(evt->getRecordAttr());

    // Note that the above means that the same record renderers are 
    // uploading mutiple things from different NPY.
    // For this to work the counts must match.
    //
    // This is necessary for the photon records and the selection index.
    //
    // All renderers ready to roll so can live switch between them, 
    // data is not duplicated thanks to Device register of uploads
}


void Scene::uploadEventSelection(OpticksEvent* evt)
{
    assert(evt);

    if(m_photon_renderer)
        m_photon_renderer->upload(evt->getSequenceAttr());

    if(m_photon_renderer)
        m_photon_renderer->upload(evt->getPhoselAttr());

    uploadRecordAttr(evt->getRecselAttr()); 
}


void Scene::uploadRecordAttr(MultiViewNPY* attr, bool debug)
{
    if(!attr) return ;  
    //assert(attr);

    if(m_record_renderer)
        m_record_renderer->upload(attr, debug);
    if(m_altrecord_renderer)
        m_altrecord_renderer->upload(attr, debug);
    if(m_devrecord_renderer)
        m_devrecord_renderer->upload(attr, debug);
}

void Scene::dump_uploads_table(const char* msg)
{
    LOG(info) << msg ; 

    if(m_photon_renderer)
        m_photon_renderer->dump_uploads_table("photon");

    if(m_source_renderer)
        m_source_renderer->dump_uploads_table("source");

    if(m_record_renderer)
        m_record_renderer->dump_uploads_table("record");

    if(m_altrecord_renderer)
        m_altrecord_renderer->dump_uploads_table("altrecord");

    if(m_devrecord_renderer)
        m_devrecord_renderer->dump_uploads_table("devrecord");
}



void Scene::preRenderCompute()
{
    for(unsigned int i=0; i<m_num_instance_renderer; i++)
    {
        if(m_instance_mode[i] && m_instance_renderer[i] && m_instance_renderer[i]->isInstLODCullEnabled() ) 
        {
            m_instance_renderer[i]->cull() ;
        }
    }
}

void Scene::renderGeometry()
{
    if(m_skipgeo_style == NOSKIPGEO )
    {
        if(*m_global_mode_ptr && m_global_renderer)       m_global_renderer->render();
        if(*m_globalvec_mode_ptr && m_globalvec_renderer) m_globalvec_renderer->render();
        // hmm this could be doing both global and globalvec ? Or does it need to be so ?


        for(unsigned int i=0; i<m_num_instance_renderer; i++)
        {
            if(m_instance_mode[i] && m_instance_renderer[i]) m_instance_renderer[i]->render();
            if(m_bbox_mode[i] && m_bbox_renderer[i])         m_bbox_renderer[i]->render();
        }
    }

    if(m_axis_mode && m_axis_renderer)     m_axis_renderer->render();
}


void Scene::renderEvent()
{
    if(m_skipevt_style == NOSKIPEVT )
    {
        if(m_genstep_mode && m_genstep_renderer)  m_genstep_renderer->render();  
        if(m_nopstep_mode && m_nopstep_renderer)  m_nopstep_renderer->render();  
        if(m_photon_mode  && m_photon_renderer)   m_photon_renderer->render();
        if(m_source_mode  && m_source_renderer)   m_source_renderer->render();
        if(m_record_mode)
        {
            Rdr* rdr = getRecordRenderer();
            if(rdr)
                rdr->render();
        }
    }
}


std::string Scene::desc() const 
{
    bool raytraced = m_composition->isRaytracedRender() ;
    bool composite = m_composition->isCompositeRender() ;
    std::stringstream ss ; 
    ss 
       << " Scene.render_count " << m_render_count
       << ( raytraced ? " raytraced " : " " )
       << ( composite ? " composite " : " " )
       << " RenderMode "  << getRenderMode() 
       ;

    return ss.str();
}

void Scene::render()
{
    //LOG(info) << desc() ; 
    m_composition->update();  // Oct 2018, moved prior to raytrace render

    bool raytraced = m_composition->isRaytracedRender() ;
    bool composite = m_composition->isCompositeRender() ;
    bool norasterized = m_composition->hasNoRasterizedRender() ;

    if(raytraced || composite)
    {
        if(m_raytrace_renderer)
            m_raytrace_renderer->render() ;
  
        if(raytraced) return ; 
        if(composite && norasterized) return ;  // didnt fix notes/issues/equirectangular_camera_blackholes_sensitive_to_far.rst
    }




    const glm::vec4& lodcut = m_composition->getLODCut();
    const glm::mat4& world2eye = m_composition->getWorld2Eye();
    const glm::mat4& world2clip = m_composition->getWorld2Clip(); 
    m_context->update( world2clip, world2eye , lodcut ); 


/*
    if(m_render_count < 1)
    {
        m_composition->Details("Scene::render.1st");
        m_composition->dumpFrustum("Scene::render.1st");
        m_composition->dumpCorners("Scene::render.1st");
    }
*/

    preRenderCompute();
    renderGeometry();
    renderEvent();

    m_render_count++ ; 
}


int Scene::touch(int ix, int iy, float depth)
{
    glm::vec3 t = m_composition->unProject(ix,iy, depth);
    gfloat3 gt(t.x, t.y, t.z );


    if(m_mesh0 == NULL)
    {
         LOG(fatal) << "Scene::touch"
                    << " mesh0 NULL "
                    ;
         return 0 ;
    }


    int container = m_mesh0->findContainer(gt);
    LOG(debug)<<"Scene::touch " 
             << " x " << t.x 
             << " y " << t.y 
             << " z " << t.z 
             << " container " << container
             ;

   if(container > 0) setTouch(container);
   return container ; 
}



void Scene::jump()
{
   // hmm what about instanced ?
    unsigned target = m_hub->getTarget();
    if( m_touch > 0 && m_touch != target )
    {
        LOG(info)<<"Scene::jump-ing from  target -> m_touch  " << target << " -> " << m_touch  ;  
        m_hub->setTarget(m_touch);
    }
}

void Scene::setTarget(unsigned int target, bool aim)
{
    m_hub->setTarget(target, aim); // sets center_extent in Composition via okg-/OpticksHub/OpticksGeometry
}
unsigned int Scene::getTarget()
{
    return m_hub->getTarget() ;
}


void Scene::setVerbosity(unsigned verbosity)
{
    m_verbosity = verbosity ;
}


const char* Scene::getShaderDir()
{
    return m_shader_dir ;
}
const char* Scene::getShaderInclPath()
{
    return m_shader_incl_path ;
}

void Scene::setGeometry(GGeoLib* geolib)
{
    m_geolib = geolib ;
}




void Scene::setInteractor(Interactor* interactor)
{
    m_interactor = interactor ;
}
Interactor* Scene::getInteractor()
{
    return m_interactor ; 
}







unsigned int Scene::getNumInstanceRenderer()
{
    return m_num_instance_renderer ; 
}

float Scene::getTimeFraction()
{
    return m_time_fraction ; 
}


unsigned int Scene::getTouch()
{
    return m_touch ;
}
void Scene::setTouch(unsigned int touch_)
{
    m_touch = touch_ ; 
}



Renderer* Scene::getGeometryRenderer()
{
    return m_geometry_renderer ; 
}

Renderer* Scene::getRaytraceRenderer()
{
    return m_raytrace_renderer ; 
}



Rdr* Scene::getAxisRenderer()
{
    return m_axis_renderer ; 
}
Rdr* Scene::getGenstepRenderer()
{
    return m_genstep_renderer ; 
}
Rdr* Scene::getNopstepRenderer()
{
    return m_nopstep_renderer ; 
}
Rdr* Scene::getPhotonRenderer()
{
    return m_photon_renderer ; 
}
Rdr* Scene::getSourceRenderer()
{
    return m_source_renderer ; 
}




Composition* Scene::getComposition()
{
    return m_composition ; 
}
Photons* Scene::getPhotons()
{
    return m_photons ; 
}


void Scene::setPhotons(Photons* photons)
{
    m_photons = photons ; 
}




//  P_KEY

void Scene::setRecordStyle(RecordStyle_t style)
{
    m_record_style = style ; 
    LOG(LEVEL) << getRecordStyleName() ; 
}
Scene::RecordStyle_t Scene::getRecordStyle()
{
    return m_record_style ; 
}
void Scene::nextRecordStyle()  // formerly nextPhotonStyle
{
    int next = (m_record_style + 1) % NUM_RECORD_STYLE ; 
    RecordStyle_t style = (RecordStyle_t)next ; 
    setRecordStyle(style);
}
void Scene::commandRecordStyle(const char* cmd)
{
    assert(cmd[0] == 'P'); 
    int style = (int)cmd[1] - (int)'0' ; 
    setRecordStyle( (RecordStyle_t)style ); 
}





//  MINUS_KEY

void Scene::nextSkipGeoStyle()  
{
    int next = (m_skipgeo_style + 1) % NUM_SKIPGEO_STYLE ; 
    SkipGeoStyle_t style = (SkipGeoStyle_t)next ; 
    setSkipGeoStyle(style);
}
void Scene::commandSkipGeoStyle(const char* cmd)
{
    assert(cmd[0] == '-'); 
    int style = (int)cmd[1] - (int)'0' ; 
    setSkipGeoStyle( style ); 
}
void Scene::setSkipGeoStyle(int style)
{
    m_skipgeo_style = (SkipGeoStyle_t)style ; 
}



//  EQUAL_KEY

void Scene::nextSkipEvtStyle()  
{
    int next = (m_skipevt_style + 1) % NUM_SKIPEVT_STYLE ; 
    SkipEvtStyle_t style = (SkipEvtStyle_t)next ; 
    setSkipEvtStyle(style);
}
void Scene::commandSkipEvtStyle(const char* cmd)
{
    assert(cmd[0] == '-'); 
    int style = (int)cmd[1] - (int)'0' ; 
    setSkipEvtStyle( style ); 
}
void Scene::setSkipEvtStyle(int style)
{
    m_skipevt_style = (SkipEvtStyle_t)style ; 
}






/**
Scene::command
-----------------

Part of SCtrl mechanism, this method is invoked by the 
high controller, eg OpticksViz, after the state has been 
changed, usually by calls to okc.Composition::command 

**/

void Scene::command(const char* cmd)
{
    assert( strlen(cmd) == 2 ); 
    switch( cmd[0] )
    {
        case 'I': commandInstanceStyle(cmd) ; break ; 
        case 'P': commandRecordStyle(cmd)   ; break ; 
        case 'B': applyContentStyle()       ; break ; 
        default:                            ; break ; 
    }
}


////// ContentStyle (B-key) /////////////////////
//// mostly moved down to okc.ContentStyle

void Scene::applyContentStyle()
{
   bool inst = m_content_style->isInst(); 
   bool bbox = m_content_style->isBBox(); 
   bool wire = m_content_style->isWire(); 
   bool asis = m_content_style->isASIS(); 
  
   if(!asis)
   {
       for(unsigned int i=0 ; i < m_num_instance_renderer ; i++ ) 
       {
           m_instance_mode[i] = inst ; 
           m_bbox_mode[i] = bbox ; 
       }
       setWireframe(wire);
   }

   LOG(debug) << "Scene::applyContentStyle" ; 
}




///  InstanceStyle(I key)

void Scene::nextInstanceStyle()
{
    int next = (m_instance_style + 1) % NUM_INSTANCE_STYLE ; 
    setInstanceStyle(next) ; 
}
void Scene::commandInstanceStyle(const char* cmd)
{
    assert( strlen(cmd) == 2 && cmd[0] == 'I' ); 
    int style = (int)cmd[1] - (int)'0' ; 
    setInstanceStyle(style) ; 
}
void Scene::setInstanceStyle(int style)
{
    m_instance_style = (InstanceStyle_t)style ; 
    applyInstanceStyle();
}
void Scene::applyInstanceStyle()  // I:key 
{
    // hmm some overlap with ContentStyle ... but that includes wireframe which can be very slow
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
   } 
}


