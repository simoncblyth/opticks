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
#include "Interactor.hh"
#include "Frame.hh"
#include "Rdr.hh"
#include "Scene.hh"
#include "AxisApp.hh"

#include "PLOG.hh"


AxisApp::AxisApp(int argc, char** argv)
        :
         m_opticks(new Opticks(argc, argv)),
         m_hub(new OpticksHub(m_opticks)),
         m_composition(NULL),
         m_scene(NULL), 
         m_frame(NULL),
         m_interactor(NULL),
         m_window(NULL),
         m_axis_renderer(NULL),
         m_axis_attr(NULL),
         m_axis_data(NULL),
         m_launcher(NULL)
{
   init();
}


MultiViewNPY* AxisApp::getAxisAttr()
{
   return m_axis_attr ; 
}

NPY<float>* AxisApp::getAxisData()
{
   return m_axis_data ; 
}

void AxisApp::setLauncher(SLauncher* launcher)
{
    m_launcher = launcher ; 
}

void AxisApp::init()
{
    m_hub->configure();
    initViz();
    prepareViz();
    upload();
}

void AxisApp::initViz()
{
   // hmm maybe OpticksViz should be at lower level, so can use here ?

    m_composition = new Composition ; 
    m_scene = new Scene ; 
    m_frame = new Frame ; 

    m_interactor = new Interactor(m_hub) ; 

    m_interactor->setFrame(m_frame);
    m_interactor->setScene(m_scene);
    m_interactor->setComposition(m_composition);
    
    m_scene->setInteractor(m_interactor);

    m_frame->setInteractor(m_interactor);
    m_frame->setComposition(m_composition);
    m_frame->setScene(m_scene);

    m_frame->setTitle("AxisApp");
}

void AxisApp::prepareViz()
{
    glm::uvec4 size = m_opticks->getSize();
    glm::uvec4 position = m_opticks->getPosition() ;

    m_composition->setSize( size );
    m_composition->setFramePosition( position );

    m_scene->setRenderMode("+axis"); 

    LOG(info) << "AxisApp::prepareViz initRenderers " ; 

    m_scene->initRenderers(); 

   // defer until renderers are setup, as distributes to them 
    m_scene->setComposition(m_composition);     
    LOG(info) << "AxisApp::prepareViz initRenderers DONE " ; 
    m_frame->init();  
    LOG(info) << "AxisApp::prepareViz frame init DONE " ; 
    m_window = m_frame->getWindow();

    LOG(info) << "AxisApp::prepareViz DONE " ; 
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

void AxisApp::render()
{
    m_frame->viewport();
    m_frame->clear();

    m_scene->render();
}

void AxisApp::renderLoop()
{
    m_frame->hintVisible(true);
    m_frame->show();
    LOG(info) << "after frame.show() "; 

    unsigned int count ; 

    while (!glfwWindowShouldClose(m_window))
    {    
        m_frame->listen(); 

        count = m_composition->tick();

        if(m_launcher)
        {
            m_launcher->launch(count);
        }

        if( m_composition->hasChanged() || m_interactor->hasChanged() || count == 1)   
        {    
            render();

            glfwSwapBuffers(m_window);

            m_interactor->setChanged(false);  
            m_composition->setChanged(false);   // sets camera, view, trackball dirty status 
        }    
    }    
}


