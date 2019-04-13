// sysrap-
#include "SLauncher.hh"

// npy-
#include "NPY.hpp"
#include "MultiViewNPY.hpp"
#include "NGLM.hpp"

// okc-
#include "Composition.hh"
#include "Opticks.hh"

// opticksgeo-
#include "OpticksHub.hh"

// oglrap-
#include "OpticksViz.hh"
#include "Interactor.hh"
#include "Frame.hh"
#include "Rdr.hh"
#include "Scene.hh"
#include "AxisApp.hh"

#include "PLOG.hh"


AxisApp::AxisApp(Opticks* ok)
    :
    m_ok(ok),
    m_hub(new OpticksHub(m_ok)),
    m_viz(new OpticksViz(m_hub, NULL)),
    m_composition(m_hub->getComposition()),
    m_scene(m_viz->getScene()), 
    m_axis_renderer(NULL),
    m_axis_attr(NULL),
    m_axis_data(NULL)
{
    init();
}

void AxisApp::init()
{
    m_viz->setTitle("AxisApp");

    m_viz->prepareScene("+axis");    // setup renderer

    upload();
}


NPY<float>* AxisApp::getAxisData()
{
    return m_axis_data ; 
}
void AxisApp::setLauncher(SLauncher* launcher)
{
    m_viz->setLauncher(launcher);
}

void AxisApp::upload()
{
    LOG(info) << "AxisApp::upload " ; 

    m_composition->update();
    m_axis_renderer = m_scene->getAxisRenderer();

    m_axis_attr = m_composition->getAxisAttr(); 
    m_axis_data = m_composition->getAxisData(); 

    bool debug = true ; 
    m_axis_renderer->upload(m_axis_attr, debug ); 

    m_scene->setTarget(0, true);
    LOG(info) << "AxisApp::upload DONE " ; 
}


void AxisApp::renderLoop()
{
    m_viz->renderLoop();
}



