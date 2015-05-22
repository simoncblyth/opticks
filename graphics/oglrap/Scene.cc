#include "Scene.hh"

#include <GL/glew.h>
#include "Composition.hh"
#include "Geometry.hh"
#include "Renderer.hh"
#include "Rdr.hh"
#include "GDrawable.hh"
#include "NumpyEvt.hpp"

#include "GLMPrint.hpp"
#include "GLMFormat.hpp"

#include <boost/lexical_cast.hpp>
#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal

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
    m_geometry_loader = new Geometry();
    m_geometry_renderer = new Renderer("nrm");
    m_genstep_renderer = new Rdr("p2l");
    m_photon_renderer = new Rdr("pos");
}

void Scene::setComposition(Composition* composition)
{
    m_composition = composition ; 
    m_geometry_renderer->setComposition(composition);
    m_genstep_renderer->setComposition(composition);
    m_photon_renderer->setComposition(composition);
}

const char* Scene::loadGeometry(const char* prefix, bool nogeocache)
{
    const char* idpath = m_geometry_loader->load(prefix, nogeocache);
    m_geometry = m_geometry_loader->getDrawable();
    m_geometry_renderer->setDrawable(m_geometry);  // upload would be better name than setDrawable

    setTarget(0);

    return idpath ;
}

void Scene::loadEvt()
{
    m_genstep_renderer->upload(m_evt->getGenstepAttr());
    m_photon_renderer->upload(m_evt->getPhotonAttr());
}

void Scene::render()
{
    m_geometry_renderer->render();
    m_genstep_renderer->render();  
    m_photon_renderer->render();
}

unsigned int Scene::touch(int ix, int iy, float depth)
{
    glm::vec3 t = m_composition->unProject(ix,iy, depth);
    gfloat3 gt(t.x, t.y, t.z );

    unsigned int container = m_geometry->findContainer(gt);
    LOG(info)<<"Scene::touch " 
             << " x " << t.x 
             << " y " << t.y 
             << " z " << t.z 
             << " container " << container
             ;

   //if(container > 0) setTarget(container);
   return container ; 
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

    LOG(info)<<"Scene::setTarget " << index << " ce " 
             << " " << ce.x 
             << " " << ce.y 
             << " " << ce.z 
             << " " << ce.w ;

    m_composition->setCenterExtent(ce); 
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

