#include "Scene.hh"

#include <GL/glew.h>

// ggeo-
#include "GLoader.hh"
#include "GDrawable.hh"

// assimpwrap
#include "AssimpWrap/AssimpGGeo.hh"

// oglrap-
#include "Composition.hh"
#include "Renderer.hh"
#include "Rdr.hh"

// npy-
#include "NumpyEvt.hpp"
#include "GLMPrint.hpp"
#include "GLMFormat.hpp"

#ifdef GUI_
#include <imgui.h>
#endif


#include <boost/lexical_cast.hpp>
#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal



const char* Scene::PHOTON = "photon" ; 
const char* Scene::GENSTEP = "genstep" ; 
const char* Scene::GEOMETRY = "geometry" ; 
const char* Scene::RECORD   = "record" ; 


void Scene::gui()
{
#ifdef GUI_
     // hmm scattering ImGui code has distinct advantages
     // means the above gymnastics to line up the choices 
     // and selection arrays is not needed
     ImGui::Checkbox(GEOMETRY, &m_geometry_mode);

     ImGui::Checkbox(GENSTEP,  &m_genstep_mode);
     ImGui::Checkbox(PHOTON,   &m_photon_mode);
     ImGui::Checkbox(RECORD,   &m_record_mode);
     ImGui::Text(" target: %u ", m_target );
     ImGui::Text(" genstep %d photon %d record %d \n", 
             m_genstep_renderer->getCountDefault(),
             m_photon_renderer->getCountDefault(),
             m_record_renderer->getCountDefault()
     );
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




void Scene::init()
{
    m_geometry_loader = new GLoader();
    m_geometry_loader->setImp(&AssimpGGeo::load);    // setting GLoaderImpFunctionPtr

    m_geometry_renderer = new Renderer("nrm");
    m_genstep_renderer = new Rdr("p2l");
    m_photon_renderer = new Rdr("pos");

    m_record_renderer = new Rdr("rec");
    m_record_renderer->setPrimitive(Rdr::LINES);

}

void Scene::setComposition(Composition* composition)
{
    m_composition = composition ; 
    m_geometry_renderer->setComposition(composition);
    m_genstep_renderer->setComposition(composition);
    m_photon_renderer->setComposition(composition);
    m_record_renderer->setComposition(composition);
}

const char* Scene::loadGeometry(const char* prefix, bool nogeocache)
{
    const char* idpath = m_geometry_loader->load(prefix, nogeocache);
    m_geometry = m_geometry_loader->getDrawable();
    m_geometry_renderer->setDrawable(m_geometry);  // upload would be better name than setDrawable

    setTarget(0);

    return idpath ;
}

void Scene::uploadEvt()
{
    m_genstep_renderer->upload(m_evt->getGenstepAttr());
    m_photon_renderer->upload(m_evt->getPhotonAttr());
    m_record_renderer->upload(m_evt->getRecordAttr());
}

void Scene::render()
{
    if(m_geometry_mode) m_geometry_renderer->render();
    if(m_genstep_mode)  m_genstep_renderer->render();  
    if(m_photon_mode)   m_photon_renderer->render();
    if(m_record_mode)   m_record_renderer->render();
}

unsigned int Scene::touch(int ix, int iy, float depth)
{
    glm::vec3 t = m_composition->unProject(ix,iy, depth);
    gfloat3 gt(t.x, t.y, t.z );

    unsigned int container = m_geometry->findContainer(gt);
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
    if( m_touch > 0 && m_touch != m_target )
    {
        LOG(info)<<"Scene::jump-ing from  m_target -> m_touch  " << m_target << " -> " << m_touch  ;  
        setTarget(m_touch);
    }
}


void Scene::setTarget(unsigned int index)
{
    m_target = index ; 

    if( m_geometry == NULL )
    {
        LOG(fatal)<<"Scene::setTarget " << index << " finds no geometry : cannot set target  " ; 
        return ;  
    }

    gfloat4 ce = m_geometry->getCenterExtent(index);

    bool autocam = true ; 
    LOG(info)<<"Scene::setTarget " << index << " ce " 
             << " " << ce.x 
             << " " << ce.y 
             << " " << ce.z 
             << " " << ce.w 
             << " autocam " << autocam 
             ;

    m_composition->setCenterExtent(ce, autocam); 
}

GMergedMesh* Scene::getMergedMesh()
{
    return m_geometry_loader ? m_geometry_loader->getMergedMesh() : NULL ;
}
GGeo* Scene::getGGeo()
{
    return m_geometry_loader ? m_geometry_loader->getGGeo() : NULL ;
}
GSubstanceLibMetadata* Scene::getMetadata()
{
    return m_geometry_loader ? m_geometry_loader->getMetadata() : NULL ;
}

