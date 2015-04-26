#include "Scene.hh"

#include <GL/glew.h>
#include "Composition.hh"
#include "Geometry.hh"
#include "Renderer.hh"
#include "Rdr.hh"
#include "GDrawable.hh"
#include "NumpyEvt.hpp"


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

void Scene::configure(const char* name, const char* value_)
{
    int value = boost::lexical_cast<int>(value_); 
    configure(name, value);
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



void Scene::configureI(const char* name, std::vector<int> values)
{
    LOG(info) << "Scene::configureI";
    if(values.empty()) return ;

    if(strcmp(name, TARGET) == 0)
    {
        int last = values.back();
        setTarget(last);
    }
}



void Scene::setComposition(Composition* composition)
{
    m_composition = composition ; 
    m_geometry_renderer->setComposition(composition);
    m_genstep_renderer->setComposition(composition);
    m_photon_renderer->setComposition(composition);
}


void Scene::loadGeometry(const char* prefix)
{
    m_geometry_loader->load(prefix);
    m_geometry = m_geometry_loader->getDrawable();
    m_geometry_renderer->setDrawable(m_geometry);  // upload would be better name than setDrawable

    setTarget(0);
}


void Scene::setTarget(unsigned int index)
{
    m_target = index ; 

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

void Scene::loadEvt()
{
    m_genstep_renderer->upload(m_evt->getGenstepAttr());
    m_photon_renderer->upload(m_evt->getPhotonAttr());
}

void Scene::render()
{
    m_geometry_renderer->render();
    m_genstep_renderer->render();   // no-show after switch to OptiX and back 
    //m_photon_renderer->render();
}


