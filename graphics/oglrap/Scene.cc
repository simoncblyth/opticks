#include "Scene.hh"

#include <GL/glew.h>
#include "Geometry.hh"
#include "Renderer.hh"
#include "Rdr.hh"
#include "GDrawable.hh"
#include "NumpyEvt.hpp"


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


void Scene::loadGeometry(const char* prefix)
{
    m_geometry_loader->load(prefix);
    m_geometry = m_geometry_loader->getDrawable();
    m_geometry_renderer->setDrawable(m_geometry);  // upload would be better name than setDrawable
}

void Scene::loadEvt()
{
    m_genstep_renderer->upload(m_evt->getGenstepAttr());
    m_photon_renderer->upload(m_evt->getPhotonAttr());
}

float* Scene::getTarget()
{
    // targeting viewpoint at evt data OR geometry 
    //float* target = m_evt->getGenstepAttr()["vpos"]->getModelToWorldPtr();  
    float* target = m_geometry->getModelToWorldPtr();
    return target ;
}


void Scene::render()
{
    m_geometry_renderer->render();
    m_genstep_renderer->render();   // no-show after switch to OptiX and back 
    //m_photon_renderer->render();
}


