#include "Scene.hh"

#include <GL/glew.h>
#include "string.h"

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

// npy-
#include "NumpyEvt.hpp"
#include "ViewNPY.hpp"
#include "MultiViewNPY.hpp"

#include "GLMPrint.hpp"
#include "GLMFormat.hpp"

#ifdef GUI_
#include <imgui.h>
#endif


#include <boost/lexical_cast.hpp>
#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal



const char* Scene::AXIS   = "axis" ; 
const char* Scene::PHOTON = "photon" ; 
const char* Scene::GENSTEP = "genstep" ; 
const char* Scene::GLOBAL  = "global" ; 

const char* Scene::INSTANCE0 = "instance0" ; 
const char* Scene::INSTANCE1 = "instance1" ; 
const char* Scene::INSTANCE2 = "instance2" ; 
const char* Scene::INSTANCE3 = "instance3" ; 
const char* Scene::INSTANCE4 = "instance4" ; 

const char* Scene::BBOX0     = "bbox0" ; 
const char* Scene::BBOX1     = "bbox1" ; 
const char* Scene::BBOX2     = "bbox2" ; 
const char* Scene::BBOX3     = "bbox3" ; 
const char* Scene::BBOX4     = "bbox4" ; 

const char* Scene::RECORD   = "record" ; 

const char* Scene::REC_ = "point" ;   
const char* Scene::ALTREC_ = "line" ; 
const char* Scene::DEVREC_ = "vector" ; 

const char* Scene::NORM_ = "norm" ;   
const char* Scene::BBOX_ = "bbox" ; 
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
      case NORM:return NORM_ ; break; 
      case BBOX:return BBOX_ ; break; 
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



void Scene::applyGeometryStyle()
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

   m_global_renderer->setWireframe(wire);

}











const char* Scene::getRecordStyleName()
{
   return getRecordStyleName(getRecordStyle());
}
 
void Scene::init()
{
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
     ImGui::Checkbox(PHOTON,   &m_photon_mode);
     ImGui::Checkbox(RECORD,   &m_record_mode);
     ImGui::Text(" target: %u ", m_target );
     ImGui::Text(" genstep %d photon %d record %d \n", 
             m_genstep_renderer->getCountDefault(),
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




void Scene::initRenderers()
{
    LOG(info) << "Scene::initRenderers " 
              << " shader_dir " << m_shader_dir 
              << " shader_incl_path " << m_shader_incl_path 
               ;
   
    assert(m_shader_dir);

    m_device = new Device();

    m_colors = new Colors(m_device);

    m_global_renderer = new Renderer("nrm", m_shader_dir, m_shader_incl_path );
    m_globalvec_renderer = new Renderer("nrmvec", m_shader_dir, m_shader_incl_path );

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

    // set for all instance slots, otherwise requires setComposition after uploadGeometry
    // as only then is m_num_instance_renderer set
    for( unsigned int i=0 ; i < MAX_INSTANCE_RENDERER ; i++)    
    {
        m_instance_renderer[i]->setComposition(composition);
        m_bbox_renderer[i]->setComposition(composition);
    }

    m_axis_renderer->setComposition(composition);
    m_genstep_renderer->setComposition(composition);
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



    LOG(info) << "Scene::uploadGeometry"
              << " nmm " << nmm
              ;

    unsigned int n_global(0);

    for(unsigned int i=0 ; i < nmm ; i++)
    {
        GMergedMesh* mm = m_ggeo->getMergedMesh(i);
        GBuffer* tbuf = mm->getTransformsBuffer();

        LOG(debug) ;
        LOG(debug) << "Scene::uploadGeometry i " << i ; 

        if( tbuf == NULL )
        {
            if(m_mesh0 == NULL) m_mesh0 = mm ; // first non-instanced mesh

            m_global_renderer->upload(mm);  
            m_globalvec_renderer->upload(mm);   // buffers are not re-uploaded, but binding must be done for each renderer 
            n_global++ ; 
            assert(n_global == 1);
            m_global_mode = true ; 
        }
        else
        {
            assert(m_num_instance_renderer < MAX_INSTANCE_RENDERER) ;

            LOG(debug)<< "Scene::uploadGeometry instance renderer " << m_num_instance_renderer  ;
            m_instance_renderer[m_num_instance_renderer]->upload(mm);
            m_instance_mode[m_num_instance_renderer] = true ; 

            LOG(debug)<< "Scene::uploadGeometry bbox renderer " << m_num_instance_renderer  ;
            GBBoxMesh* bb = GBBoxMesh::create(mm); assert(bb);
            m_bbox_mode[m_num_instance_renderer] = true ; 
            m_bbox_renderer[m_num_instance_renderer]->upload(bb);

            m_num_instance_renderer++ ; 
        }
    }


    LOG(info)<<"Scene::uploadGeometry" 
             << " n_global "   << n_global
             << " m_num_instance_renderer " << m_num_instance_renderer
             ; 

    applyGeometryStyle(); // sets m_instance_mode m_bbox_mode switches, change with "B"  nextGeometryStyle()
}

void Scene::uploadColorBuffer(GBuffer* colorbuffer)
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



void Scene::uploadAxis()
{
    LOG(info) << "Scene::uploadAxis  " ;
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

    m_photon_renderer->upload(m_evt->getPhotonAttr());
    m_photon_renderer->upload(m_evt->getSequenceAttr());
    m_photon_renderer->upload(m_evt->getPhoselAttr());

    uploadRecordAttr(m_evt->getRecordAttr());

    uploadRecordAttr(m_evt->getAuxAttr());
}


void Scene::uploadSelection()
{
    // this was used after the slow SequenceNPY (CPU side std::map based photon history/material indexing)
    // following move to thrust is not done just after uploadEvt
    assert(m_evt);
    LOG(info)<<"Scene::uploadSelection";
    uploadRecordAttr(m_evt->getRecselAttr()); 
}


void Scene::uploadRecordAttr(MultiViewNPY* attr)
{
    if(!attr) return ;  
    //assert(attr);

    // all renderers ready to roll so can live switch between them, 
    // data is not duplicated thanks to Device register of uploads

    m_record_renderer->upload(attr);
    m_altrecord_renderer->upload(attr);
    m_devrecord_renderer->upload(attr);
}




void Scene::render()
{
    if(m_global_mode)    m_global_renderer->render();
    if(m_globalvec_mode) m_globalvec_renderer->render();

    for(unsigned int i=0; i<m_num_instance_renderer; i++)
    {
        if(m_instance_mode[i]) m_instance_renderer[i]->render();
        if(m_bbox_mode[i])     m_bbox_renderer[i]->render();
    }

    if(m_axis_mode)     m_axis_renderer->render();
    if(m_genstep_mode)  m_genstep_renderer->render();  
    if(m_photon_mode)   m_photon_renderer->render();
    if(m_record_mode)
    {
        Rdr* rdr = getRecordRenderer();
        assert(rdr);
        rdr->render();
    }
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


void Scene::setTarget(unsigned int target, bool autocam)
{
    m_target = target ; 

    gfloat4 ce = m_mesh0->getCenterExtent(target);

    LOG(info)<<"Scene::setTarget " 
             << " target " << target 
             << " autocam " << autocam 
             << " ce " 
             << " " << ce.x 
             << " " << ce.y 
             << " " << ce.z 
             << " " << ce.w 
             ;

    m_composition->setCenterExtent(ce, autocam); 
}


void Scene::setFaceTarget(unsigned int face_index, unsigned int solid_index, unsigned int mesh_index)
{
    assert(m_ggeo && "must setGeometry first");
    glm::vec4 ce = m_ggeo->getFaceCenterExtent(face_index, solid_index, mesh_index);

    bool autocam = false ; 
    m_composition->setCenterExtent(ce, autocam );
}

void Scene::setFaceRangeTarget(unsigned int face_index0, unsigned int face_index1, unsigned int solid_index, unsigned int mesh_index)
{
    assert(m_ggeo && "must setGeometry first");
    glm::vec4 ce = m_ggeo->getFaceRangeCenterExtent(face_index0, face_index1, solid_index, mesh_index);

    bool autocam = false ; 
    m_composition->setCenterExtent(ce, autocam );
}


