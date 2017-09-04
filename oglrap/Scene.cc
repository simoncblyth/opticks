#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cstring>
#include <string>
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
#include "OpticksConst.hh"
#include "OpticksEvent.hh"

// okg-
#include "OpticksHub.hh"

// ggeo-
#include "GMergedMesh.hh"
#include "GBBoxMesh.hh"
#include "GGeoLib.hh"


// oglrap-
#include "Config.hh"      // cmake generated header

#include "Composition.hh"
#include "Renderer.hh"
#include "InstLODCull.hh"
#include "Device.hh"
#include "Rdr.hh"
#include "Colors.hh"
#include "Interactor.hh"

#ifdef GUI_
#include <ImGui/imgui.h>
#endif



#include "PLOG.hh"


const char* Scene::PREFIX = "scene" ;
const char* Scene::getPrefix()
{
   return PREFIX ; 
}




const char* Scene::AXIS   = "axis" ; 
const char* Scene::PHOTON = "photon" ; 
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

const char* Scene::BBOX_ = "bbox" ; 
const char* Scene::NORM_ = "norm" ;   
const char* Scene::NONE_ = "none" ;   
const char* Scene::WIRE_ = "wire" ; 
const char* Scene::NORM_BBOX_ = "norm_bbox" ; 



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
 
void Scene::init()
{
    LOG(debug) << "Scene::init" ; 
    LOG(trace) << "Scene::init (config from cmake)"
              << " OGLRAP_INSTALL_PREFIX " << OGLRAP_INSTALL_PREFIX
              << " OGLRAP_SHADER_DIR " << OGLRAP_SHADER_DIR
              << " OGLRAP_SHADER_INCL_PATH " << OGLRAP_SHADER_INCL_PATH
              << " OGLRAP_SHADER_DYNAMIC_DIR " << OGLRAP_SHADER_DYNAMIC_DIR
              ;   


    if(m_shader_dir == NULL)
    {
        m_shader_dir = strdup(OGLRAP_SHADER_DIR);
    }
    if(m_shader_incl_path == NULL)
    {
        m_shader_incl_path = strdup(OGLRAP_SHADER_INCL_PATH);
    }
    if(m_shader_dynamic_dir == NULL)
    {
        m_shader_dynamic_dir = strdup(OGLRAP_SHADER_DYNAMIC_DIR);
    }
}

void Scene::write(BDynamicDefine* dd)
{
    dd->write( m_shader_dynamic_dir, "dynamic.h" );
}

void Scene::setRenderMode(const char* s)
{
    // setting renderer toggles

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
             unsigned int ins = boost::lexical_cast<unsigned int>(el+strlen(_INSTANCE)) ;
             if(ins < MAX_INSTANCE_RENDERER)
                  *(m_instance_mode+ins) = setting ;  
        } 
        
        if(strcmp(el, GLOBAL)==0)  m_global_mode = setting ; 
        if(strcmp(el, AXIS)==0)    m_axis_mode = setting ; 
        if(strcmp(el, GENSTEP)==0) m_genstep_mode = setting ; 
        if(strcmp(el, NOPSTEP)==0) m_nopstep_mode = setting ; 
        if(strcmp(el, PHOTON)==0)  m_photon_mode = setting ; 
        if(strcmp(el, RECORD)==0)  m_record_mode = setting ; 
    }
}

std::string Scene::getRenderMode()
{
    const char* delim = "," ; 

    std::stringstream ss ; 

    if(m_global_mode)  ss << GLOBAL << delim ; 
    if(m_axis_mode)    ss << AXIS << delim ; 
    if(m_genstep_mode) ss << GENSTEP << delim ; 
    if(m_nopstep_mode) ss << NOPSTEP << delim ; 
    if(m_photon_mode) ss << PHOTON << delim ; 
    if(m_record_mode) ss << RECORD << delim ; 

    for(unsigned int i=0 ; i<MAX_INSTANCE_RENDERER ; i++) if(m_instance_mode[i]) ss << _INSTANCE << i << delim ; 
    for(unsigned int i=0 ; i<MAX_INSTANCE_RENDERER ; i++) if(m_bbox_mode[i]) ss << _BBOX << i << delim ; 

    return ss.str();
}





void Scene::gui()
{
#ifdef GUI_
     ImGui::Checkbox(GLOBAL,   &m_global_mode);

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
     ImGui::Checkbox(RECORD,   &m_record_mode);
    // ImGui::Text(" target: %u ", m_target );
     ImGui::Text(" genstep %d nopstep %d photon %d record %d \n", 
             ( m_genstep_renderer ? m_genstep_renderer->getCountDefault() : -1 ),
             ( m_nopstep_renderer ? m_nopstep_renderer->getCountDefault() : -1 ),
             ( m_photon_renderer ? m_photon_renderer->getCountDefault() : -1 ),
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

    m_global_renderer = new Renderer("nrm", m_shader_dir, m_shader_incl_path );
    m_globalvec_renderer = new Renderer("nrmvec", m_shader_dir, m_shader_incl_path );
    m_raytrace_renderer = new Renderer("tex", m_shader_dir, m_shader_incl_path );

   // small array of instance renderers to handle multiple assemblies of repeats 
    for( unsigned int i=0 ; i < MAX_INSTANCE_RENDERER ; i++)
    {
        m_instance_mode[i] = false ; 

        m_instance_renderer[i] = new Renderer("inrm", m_shader_dir, m_shader_incl_path );
        m_instance_renderer[i]->setInstanced();

        if(m_instcull)
        {
            m_instlodcull[i] = new InstLODCull("inrmcull", m_shader_dir, m_shader_incl_path );
            m_instlodcull[i]->setVerbosity(1);
            m_instance_renderer[i]->setInstLODCull(m_instlodcull[i]);
        }

        m_bbox_mode[i] = false ; 
        m_bbox_renderer[i] = new Renderer("inrm", m_shader_dir, m_shader_incl_path );
        m_bbox_renderer[i]->setInstanced();
        m_bbox_renderer[i]->setWireframe(false);  // wireframe is much slower than filled
    }

    //LOG(info) << "Scene::init geometry_renderer ctor DONE";

    m_axis_renderer = new Rdr(m_device, "axis", m_shader_dir, m_shader_incl_path );

    m_genstep_renderer = new Rdr(m_device, "p2l", m_shader_dir, m_shader_incl_path);

    m_nopstep_renderer = new Rdr(m_device, "nop", m_shader_dir, m_shader_incl_path);
    m_nopstep_renderer->setPrimitive(Rdr::LINE_STRIP);

    m_photon_renderer = new Rdr(m_device, "pos", m_shader_dir, m_shader_incl_path );

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

void Scene::setComposition(Composition* composition)
{
    // HMM better to use hub ? to avoid this lot ?

    m_composition = composition ; 

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

        if(m_instlodcull[i])
            m_instlodcull[i]->setComposition(composition);

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

    if(m_record_renderer)
        m_record_renderer->setComposition(composition);

    if(m_altrecord_renderer)
        m_altrecord_renderer->setComposition(composition);

    if(m_devrecord_renderer)
        m_devrecord_renderer->setComposition(composition);
}



void Scene::uploadGeometryGlobal(GMergedMesh* mm)
{
    LOG(debug)<< "Scene::uploadGeometryGlobal " ;

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
        m_global_mode = true ;
    }
    else
    {
         LOG(warning) << "Scene::uploadGeometryGlobal SKIPPING GLOBAL " ; 
    }
}


void Scene::uploadGeometryInstanced(GMergedMesh* mm)
{
    bool empty = mm->isEmpty();
    bool skip = mm->isSkip() ;

    if(!skip && !empty)
    { 
        assert(m_num_instance_renderer < MAX_INSTANCE_RENDERER) ;
        LOG(info)<< "Scene::uploadGeometryInstanced instance renderer " << m_num_instance_renderer << " instcull " << m_instcull ;

        NPY<float>* ibuf = mm->getITransformsBuffer();
        assert(ibuf);

        if(m_instance_renderer[m_num_instance_renderer])
        {
            m_instance_renderer[m_num_instance_renderer]->setLOD(m_ok->getLOD());
            m_instance_renderer[m_num_instance_renderer]->upload(mm);
            m_instance_mode[m_num_instance_renderer] = true ; 

        }

        LOG(trace)<< "Scene::uploadGeometryInstanced bbox renderer " << m_num_instance_renderer  ;
        GBBoxMesh* bb = GBBoxMesh::create(mm); assert(bb);

        if(m_bbox_renderer[m_num_instance_renderer])
        {
            m_bbox_renderer[m_num_instance_renderer]->upload(bb);
            m_bbox_mode[m_num_instance_renderer] = true ; 
        }
        m_num_instance_renderer++ ; 
    }
    else
    {
         LOG(warning) << "Scene::uploadGeometry SKIPPING " 
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

    LOG(info) << "Scene::uploadGeometry"
              << " nmm " << nmm
              ;

    for(unsigned int i=0 ; i < nmm ; i++)
    {
        GMergedMesh* mm = m_geolib->getMergedMesh(i);
        if(!mm) continue ; 

        LOG(info) << "Scene::uploadGeometry " 
                  << i 
                  << " geoCode " << mm->getGeoCode() ; 

        if( i == 0 )  // first mesh assumed to be **the one and only** non-instanced global mesh
        {
           uploadGeometryGlobal(mm);
        }
        else
        {
           uploadGeometryInstanced(mm);
        }
    }

    LOG(debug)<<"Scene::uploadGeometry" 
             << " m_num_instance_renderer " << m_num_instance_renderer
             ; 

    applyGeometryStyle(); // sets m_instance_mode m_bbox_mode switches, change with "B"  nextGeometryStyle()
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
    LOG(debug) << "Scene::upload START  " ;
        
    uploadAxis();

    LOG(debug) << "Scene::upload uploadAxis  DONE " ;

    uploadEvent(evt);  // Scene, Rdr uploads orchestrated by OpticksEvent/MultiViewNPY

    LOG(debug) << "Scene::upload uploadEvt  DONE " ;

    uploadEventSelection(evt);   // recsel upload

    LOG(debug) << "Scene::upload uploadSelection  DONE " ;

    LOG(debug) << "Scene::upload DONE  " ;
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
       LOG(fatal) << "Scene::uploadEvt no evt " ;
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

    if(m_record_renderer)
        m_record_renderer->dump_uploads_table("record");

    if(m_altrecord_renderer)
        m_altrecord_renderer->dump_uploads_table("altrecord");

    if(m_devrecord_renderer)
        m_devrecord_renderer->dump_uploads_table("devrecord");
}


void Scene::renderGeometry()
{
    if(m_global_mode && m_global_renderer)       m_global_renderer->render();
    if(m_globalvec_mode && m_globalvec_renderer) m_globalvec_renderer->render();

    for(unsigned int i=0; i<m_num_instance_renderer; i++)
    {
        if(m_instance_mode[i] && m_instance_renderer[i]) m_instance_renderer[i]->render();
        if(m_bbox_mode[i] && m_bbox_renderer[i])         m_bbox_renderer[i]->render();
    }
    if(m_axis_mode && m_axis_renderer)     m_axis_renderer->render();
}


void Scene::renderEvent()
{
    if(m_genstep_mode && m_genstep_renderer)  m_genstep_renderer->render();  
    if(m_nopstep_mode && m_nopstep_renderer)  m_nopstep_renderer->render();  
    if(m_photon_mode  && m_photon_renderer)   m_photon_renderer->render();
    if(m_record_mode)
    {
        Rdr* rdr = getRecordRenderer();
        if(rdr)
            rdr->render();
    }
}

void Scene::render()
{
    bool raytraced = isRaytracedRender() ;
    bool composite = isCompositeRender() ;

    if(raytraced || composite)
    {
        if(m_raytrace_renderer)
            m_raytrace_renderer->render() ;
        if(raytraced) return ; 
    }

    m_composition->update();

/*
    if(m_render_count < 1)
    {
        m_composition->Details("Scene::render.1st");
        m_composition->dumpFrustum("Scene::render.1st");
        m_composition->dumpCorners("Scene::render.1st");
    }
*/

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





void Scene::nextRenderStyle(unsigned int modifiers)  // O:key cycling: Projective, Raytraced, Composite 
{
    bool nudge = modifiers & OpticksConst::e_shift ;
    if(nudge)
    {
        m_composition->setChanged(true) ;
        return ; 
    }

    int next = (m_render_style + 1) % NUM_RENDER_STYLE ; 
    m_render_style = (RenderStyle_t)next ; 
    applyRenderStyle();

    m_composition->setChanged(true) ; // trying to avoid the need for shift-O nudging 
}

bool Scene::isProjectiveRender()
{
   return m_render_style == R_PROJECTIVE ;
}
bool Scene::isRaytracedRender()
{
   return m_render_style == R_RAYTRACED ;
}
bool Scene::isCompositeRender()
{
   return m_render_style == R_COMPOSITE ;
}

void Scene::applyRenderStyle()   
{
    // nothing to do, style is honoured by  Scene::render
}


Scene::Scene(OpticksHub* hub, const char* shader_dir, const char* shader_incl_path, const char* shader_dynamic_dir) 
            :
            m_hub(hub),
            m_ok(hub->getOpticks()),
            m_shader_dir(shader_dir ? strdup(shader_dir): NULL ),
            m_shader_dynamic_dir(shader_dynamic_dir ? strdup(shader_dynamic_dir): NULL),
            m_shader_incl_path(shader_incl_path ? strdup(shader_incl_path): NULL),
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
            m_record_renderer(NULL),
            m_altrecord_renderer(NULL),
            m_devrecord_renderer(NULL),
            m_photons(NULL),
            m_geolib(NULL),
            m_mesh0(NULL),
            m_composition(NULL),
            m_colorbuffer(NULL),
            m_touch(0),
            m_global_mode(false),
            m_globalvec_mode(false),
            m_axis_mode(true),
            m_genstep_mode(true),
            m_nopstep_mode(true),
            m_photon_mode(true),
            m_record_mode(true),
            m_record_style(ALTREC),
            m_geometry_style(BBOX),
            m_num_geometry_style(0),
            m_global_style(GVIS),
            m_num_global_style(0),
            m_instance_style(IVIS),
            m_render_style(R_PROJECTIVE),
            m_initialized(false),
            m_time_fraction(0.f),
            m_instcull(true),
            m_verbosity(0),
            m_render_count(0)
{

    init();

    for(unsigned int i=0 ; i < MAX_INSTANCE_RENDERER ; i++ ) 
    {
        m_instance_renderer[i] = NULL ; 
        m_instlodcull[i] = NULL ; 
        m_bbox_renderer[i] = NULL ; 
        m_instance_mode[i] = false ; 
        m_bbox_mode[i] = false ; 
    }
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




void Scene::setRecordStyle(RecordStyle_t style)
{
    m_record_style = style ; 
}

Scene::RecordStyle_t Scene::getRecordStyle()
{
    return m_record_style ; 
}







void Scene::nextPhotonStyle()
{
    int next = (m_record_style + 1) % NUM_RECORD_STYLE ; 
    m_record_style = (RecordStyle_t)next ; 
}




////// GeometryStyle (B-key) /////////////////////

unsigned int Scene::getNumGeometryStyle()
{
    return m_num_geometry_style == 0 ? int(NUM_GEOMETRY_STYLE) : m_num_geometry_style ;
}
void Scene::setNumGeometryStyle(unsigned int num_geometry_style)
{
    m_num_geometry_style = num_geometry_style ;

    dumpGeometryStyles("Scene::setNumGeometryStyle");
}

void Scene::dumpGeometryStyles(const char* msg)
{
    const GeometryStyle_t style0 = getGeometryStyle() ;
    GeometryStyle_t style = style0 ; 

    LOG(info) << msg << " (Scene::dumpGeometryStyles) " ; 

    while( style != style0 )
    {
        nextGeometryStyle();
        style = getGeometryStyle() ;
    }
 
    assert( style == style0 );
}

Scene::GeometryStyle_t Scene::getGeometryStyle() const 
{
    return m_geometry_style ;
}

void Scene::nextGeometryStyle()
{
    unsigned num_geometry_style = getNumGeometryStyle() ;
    int next = (m_geometry_style + 1) % num_geometry_style ; 
    setGeometryStyle( (GeometryStyle_t)next );

    const char* stylename = getGeometryStyleName();
    printf("Scene::nextGeometryStyle : %s num_geometry_style %u m_num_geometry_style %u  \n", stylename, num_geometry_style, m_num_geometry_style );
}

void Scene::setGeometryStyle(GeometryStyle_t style)
{
    m_geometry_style = style ; 
    applyGeometryStyle();
}

const char* Scene::getGeometryStyleName(Scene::GeometryStyle_t style)
{
   switch(style)
   {
      case BBOX:return BBOX_ ; break; 
      case NORM:return NORM_ ; break; 
      case NONE:return NONE_ ; break; 
      case WIRE:return WIRE_ ; break; 
      case NORM_BBOX:return NORM_BBOX_ ; break; 
      case NUM_GEOMETRY_STYLE:assert(0) ; break ; 
      default: assert(0); break ; 
   } 
   return NULL ; 
}

const char* Scene::getGeometryStyleName()
{
   return getGeometryStyleName(m_geometry_style);
}

void Scene::applyGeometryStyle()  // B:key 
{
    bool inst(false) ; 
    bool bbox(false) ; 
    bool wire(false) ; 

    switch(m_geometry_style)
    {
      case BBOX:
             inst = false ; 
             bbox = true ; 
             wire = false ; 
             break;
      case NORM:
             inst = true ;
             bbox = false ; 
             wire = false ; 
             break;
      case NONE:
             inst = false ;
             bbox = false ; 
             wire = false ; 
             break;
      case WIRE:
             inst = true ;
             bbox = false ; 
             wire = true ; 
             break;
      case NORM_BBOX:
             inst = true ; 
             bbox = true ; 
             wire = false ; 
             break;
      case NUM_GEOMETRY_STYLE:
             assert(0);
             break;
   }

   for(unsigned int i=0 ; i < m_num_instance_renderer ; i++ ) 
   {
       m_instance_mode[i] = inst ; 
       m_bbox_mode[i] = bbox ; 
   }

   setWireframe(wire);
}







// GlobalStyle (Q key)

unsigned int Scene::getNumGlobalStyle()
{
    return m_num_global_style == 0 ? int(NUM_GLOBAL_STYLE) : m_num_global_style ;
}
void Scene::setNumGlobalStyle(unsigned int num_global_style)
{
    m_num_global_style = num_global_style ;
}

void Scene::nextGlobalStyle()
{
    int next = (m_global_style + 1) % getNumGlobalStyle() ; 
    m_global_style = (GlobalStyle_t)next ; 
    applyGlobalStyle();
}

void Scene::applyGlobalStyle()
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






///  InstanceStyle(I key)



void Scene::nextInstanceStyle()
{
    int next = (m_instance_style + 1) % NUM_INSTANCE_STYLE ; 
    m_instance_style = (InstanceStyle_t)next ; 
    applyInstanceStyle();
}

void Scene::applyInstanceStyle()  // I:key 
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


