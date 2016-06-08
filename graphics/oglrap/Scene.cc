#include "Scene.hh"

#include <GL/glew.h>
#include <cstring>

// opticks-
#include "Opticks.hh"
#include "OpticksEvent.hh"

// ggeo-
#include "GMergedMesh.hh"
#include "GBBoxMesh.hh"
#include "GGeo.hh"


// oglrap-
#include "Config.hh"      // cmake generated header
#include "DynamicDefine.hh"

#include "Composition.hh"
#include "Renderer.hh"
#include "Device.hh"
#include "Rdr.hh"
#include "Colors.hh"
#include "Interactor.hh"

// npy-
#include "NPY.hpp"
#include "ViewNPY.hpp"
#include "MultiViewNPY.hpp"

#include "GLMPrint.hpp"
#include "GLMFormat.hpp"

#ifdef GUI_
#include <imgui.h>
#endif

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include "BLog.hh"


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
    bool inst ; 
    bool bbox ; 
    bool wire ; 

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











const char* Scene::getRecordStyleName()
{
   return getRecordStyleName(getRecordStyle());
}
 
void Scene::init()
{

    LOG(info) << "Scene::init (config from cmake)"
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

void Scene::write(DynamicDefine* dd)
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
     ImGui::Text(" target: %u ", m_target );
     ImGui::Text(" genstep %d nopstep %d photon %d record %d \n", 
             m_genstep_renderer->getCountDefault(),
             m_nopstep_renderer->getCountDefault(),
             m_photon_renderer->getCountDefault(),
             m_record_renderer->getCountDefault()
     );



     int* record_style = (int*)&m_record_style ;       // address of enum cast to int*
     ImGui::RadioButton("rec",    record_style, REC); 
     ImGui::SameLine();
     ImGui::RadioButton("altrec", record_style, ALTREC); 
     ImGui::SameLine();
     ImGui::RadioButton("devrec", record_style, DEVREC); 


#endif    
}

const char* Scene::TARGET = "target" ; 

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



void Scene::setWireframe(bool wire)
{
    m_global_renderer->setWireframe(wire);

    for( unsigned int i=0 ; i < MAX_INSTANCE_RENDERER ; i++)
    {
        m_instance_renderer[i]->setWireframe(wire);

        m_bbox_renderer[i]->setWireframe(false);  

        // wireframe is much slower than filled, 
        // also bbox winding order is not correct
        // so keeping the bbox as filled
    }
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

        m_bbox_mode[i] = false ; 
        m_bbox_renderer[i] = new Renderer("inrm", m_shader_dir, m_shader_incl_path );
        m_bbox_renderer[i]->setInstanced();
        m_bbox_renderer[i]->setWireframe(false);  // wireframe is much slower than filled
    }

    //LOG(info) << "Scene::init geometry_renderer ctor DONE";

    m_axis_renderer = new Rdr(m_device, "axis", m_shader_dir, m_shader_incl_path );

    m_genstep_renderer = new Rdr(m_device, "p2l", m_shader_dir, m_shader_incl_path);

    bool nopdbg = false ; 
    if(nopdbg)
    {
        m_nopstep_renderer = new Rdr(m_device, "dbg", m_shader_dir, m_shader_incl_path);
    }
    else
    {
        m_nopstep_renderer = new Rdr(m_device, "nop", m_shader_dir, m_shader_incl_path);
        m_nopstep_renderer->setPrimitive(Rdr::LINE_STRIP);
    }


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
    m_composition = composition ; 

    m_global_renderer->setComposition(composition);
    m_globalvec_renderer->setComposition(composition);
    m_raytrace_renderer->setComposition(composition);

    // set for all instance slots, otherwise requires setComposition after uploadGeometry
    // as only then is m_num_instance_renderer set
    for( unsigned int i=0 ; i < MAX_INSTANCE_RENDERER ; i++)    
    {
        m_instance_renderer[i]->setComposition(composition);
        m_bbox_renderer[i]->setComposition(composition);
    }

    m_axis_renderer->setComposition(composition);
    m_genstep_renderer->setComposition(composition);
    m_nopstep_renderer->setComposition(composition);
    m_photon_renderer->setComposition(composition);
    m_record_renderer->setComposition(composition);
    m_altrecord_renderer->setComposition(composition);
    m_devrecord_renderer->setComposition(composition);
}


void Scene::uploadGeometry()
{
    // currently invoked from ggeoview main
    assert(m_ggeo && "must setGeometry first");
    unsigned int nmm = m_ggeo->getNumMergedMesh();

    LOG(debug) << "Scene::uploadGeometry"
              << " nmm " << nmm
              ;

    unsigned int n_global(0);

    for(unsigned int i=0 ; i < nmm ; i++)
    {
        GMergedMesh* mm = m_ggeo->getMergedMesh(i);
        bool skip = mm->isSkip() ;
        LOG(debug) << "Scene::uploadGeometry " 
                  << i 
                  << " geoCode " << mm->getGeoCode() ; 

        if( i == 0 )  // first mesh assumed to be **the one and only** non-instanced global mesh
        {
            assert(m_mesh0 == NULL); // not expected to Scene::uploadGeomety more than once 
            m_mesh0 = mm ; 

            if(!skip)
            {
                m_global_renderer->upload(mm);  
                m_globalvec_renderer->upload(mm);   // buffers are not re-uploaded, but binding must be done for each renderer 
                n_global++ ; 
                assert(n_global == 1);
                m_global_mode = true ;
            }
            else
            {
                 LOG(warning) << "Scene::uploadGeometry SKIPPING GLOBAL " << i ; 
            }
        }
        else
        {
            if(!skip)
            { 

                assert(m_num_instance_renderer < MAX_INSTANCE_RENDERER) ;
                LOG(info)<< "Scene::uploadGeometry instance renderer " << m_num_instance_renderer  ;

                NPY<float>* ibuf = mm->getITransformsBuffer();
                assert(ibuf);

                m_instance_renderer[m_num_instance_renderer]->upload(mm);
                m_instance_mode[m_num_instance_renderer] = true ; 

                LOG(debug)<< "Scene::uploadGeometry bbox renderer " << m_num_instance_renderer  ;
                GBBoxMesh* bb = GBBoxMesh::create(mm); assert(bb);

                m_bbox_mode[m_num_instance_renderer] = true ; 
                m_bbox_renderer[m_num_instance_renderer]->upload(bb);

                m_num_instance_renderer++ ; 

            }
            else
            {
                 LOG(warning) << "Scene::uploadGeometry SKIPPING " << i ; 
            }

        }
    }


    LOG(debug)<<"Scene::uploadGeometry" 
             << " n_global "   << n_global
             << " m_num_instance_renderer " << m_num_instance_renderer
             ; 

    applyGeometryStyle(); // sets m_instance_mode m_bbox_mode switches, change with "B"  nextGeometryStyle()
}

void Scene::uploadColorBuffer(NPY<unsigned char>* colorbuffer)
{
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



void Scene::upload()
{
    LOG(debug) << "Scene::upload START  " ;
        
    uploadAxis();

    LOG(debug) << "Scene::upload uploadAxis  DONE " ;

    uploadEvt();  // Scene, Rdr uploads orchestrated by OpticksEvent/MultiViewNPY

    LOG(debug) << "Scene::upload uploadEvt  DONE " ;

    uploadSelection();   // recsel upload

    LOG(debug) << "Scene::upload uploadSelection  DONE " ;

    LOG(debug) << "Scene::upload DONE  " ;
}



void Scene::uploadAxis()
{
    m_axis_renderer->upload(m_composition->getAxisAttr());
}

void Scene::uploadEvt()
{
    if(!m_evt) 
    {
       LOG(fatal) << "Scene::uploadEvt no evt " ;
       assert(m_evt);
    }

    // The Rdr call glBufferData using bytes and size from the associated NPY 
    // the bytes used is NULL when npy->hasData() == false
    // corresponding to device side only OpenGL allocation

    m_genstep_renderer->upload(m_evt->getGenstepAttr());

    m_nopstep_renderer->upload(m_evt->getNopstepAttr(), false);

    m_photon_renderer->upload(m_evt->getPhotonAttr());


    uploadRecordAttr(m_evt->getRecordAttr());

    //uploadRecordAttr(m_evt->getAuxAttr());

    // Note that the above means that the same record renderers are 
    // uploading mutiple things from different NPY.
    // For this to work the counts must match.
    //
    // This is necessary for the photon records and the selection index.
    //
    // All renderers ready to roll so can live switch between them, 
    // data is not duplicated thanks to Device register of uploads
}


void Scene::uploadSelection()
{
    assert(m_evt);

    m_photon_renderer->upload(m_evt->getSequenceAttr());
    m_photon_renderer->upload(m_evt->getPhoselAttr());

    uploadRecordAttr(m_evt->getRecselAttr()); 
}


void Scene::uploadRecordAttr(MultiViewNPY* attr, bool debug)
{
    if(!attr) return ;  
    //assert(attr);

    m_record_renderer->upload(attr, debug);
    m_altrecord_renderer->upload(attr, debug);
    m_devrecord_renderer->upload(attr, debug);
}

void Scene::dump_uploads_table(const char* msg)
{
    LOG(info) << msg ; 
    m_photon_renderer->dump_uploads_table("photon");
    m_record_renderer->dump_uploads_table("record");
    m_altrecord_renderer->dump_uploads_table("altrecord");
    m_devrecord_renderer->dump_uploads_table("devrecord");
}


void Scene::renderGeometry()
{
    if(m_global_mode)    m_global_renderer->render();
    if(m_globalvec_mode) m_globalvec_renderer->render();

    for(unsigned int i=0; i<m_num_instance_renderer; i++)
    {
        if(m_instance_mode[i]) m_instance_renderer[i]->render();
        if(m_bbox_mode[i])     m_bbox_renderer[i]->render();
    }
    if(m_axis_mode)     m_axis_renderer->render();
}


void Scene::renderEvent()
{
    if(m_genstep_mode)  m_genstep_renderer->render();  
    if(m_nopstep_mode)  m_nopstep_renderer->render();  
    if(m_photon_mode)   m_photon_renderer->render();
    if(m_record_mode)
    {
        Rdr* rdr = getRecordRenderer();
        assert(rdr);
        rdr->render();
    }
}

void Scene::render()
{
    bool raytraced = isRaytracedRender() ;
    bool composite = isCompositeRender() ;

    if(raytraced || composite)
    {
        m_raytrace_renderer->render() ;
        if(raytraced) return ; 
    }

    renderGeometry();
    renderEvent();
}


unsigned int Scene::touch(int ix, int iy, float depth)
{
    glm::vec3 t = m_composition->unProject(ix,iy, depth);
    gfloat3 gt(t.x, t.y, t.z );

    unsigned int container = m_mesh0->findContainer(gt);
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
    if( m_touch > 0 && m_touch != m_target )
    {
        LOG(info)<<"Scene::jump-ing from  m_target -> m_touch  " << m_target << " -> " << m_touch  ;  
        setTarget(m_touch);
    }
}



void Scene::setTarget(unsigned int target, bool aim)
{
    if(m_mesh0 == NULL)
    {
        LOG(info) << "Scene::setTarget " << target << " deferring as geometry not loaded " ; 
        m_target_deferred = target ; 
        return ; 
    }
    m_target = target ; 

    gfloat4 ce_ = m_mesh0->getCenterExtent(target);

    glm::vec4 ce(ce_.x, ce_.y, ce_.z, ce_.w ); 

    LOG(info)<<"Scene::setTarget " 
             << " target " << target 
             << " aim " << aim
             << " ce " 
             << " " << ce.x 
             << " " << ce.y 
             << " " << ce.z 
             << " " << ce.w 
             ;

    m_composition->setCenterExtent(ce, aim); 
}




void Scene::nextRenderStyle(unsigned int modifiers)  // O:key
{
    bool nudge = modifiers & Opticks::e_shift ;
    if(nudge)
    {
        m_composition->setChanged(true) ;
        return ; 
    }

    int next = (m_render_style + 1) % NUM_RENDER_STYLE ; 
    m_render_style = (RenderStyle_t)next ; 
    applyRenderStyle();
}

void Scene::applyRenderStyle()   
{
    // nothing to do, style is honoured by  Scene::render
}



