// npy-
#include "NGLM.hpp"

// okc-
#include "Opticks.hh"

// okg-
#include "OpticksHub.hh"

// ggeo-
#include "GGeo.hh"

// oglrap-
#include "Composition.hh"
#include "Interactor.hh"
#include "Frame.hh"
#include "Scene.hh"

#include "OGLRAP_LOG.hh"
#include "PLOG.hh"


GGeo*        m_ggeo = NULL ;
GGeoLib*     m_geolib = NULL ;

Composition* m_composition = NULL ;
Scene*       m_scene = NULL ;
Frame*       m_frame = NULL ;
Interactor*  m_interactor = NULL ;
GLFWwindow*  m_window = NULL ;


void render()
{
    m_frame->viewport();
    m_frame->clear();

    m_scene->render();
}

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    OGLRAP_LOG__ ; 
    LOG(info) << argv[0] ; 

    Opticks* m_opticks = new Opticks(argc, argv);
    OpticksHub* m_hub = new OpticksHub(m_opticks);

    //  hmm below is now done inside the Hub 
    m_ggeo = new GGeo(m_opticks);
    m_ggeo->loadFromCache();
    m_ggeo->dumpStats();

    m_geolib = m_ggeo->getGeoLib();

    // see App::initViz

    m_composition = new Composition ; 

    m_scene = new Scene(m_hub) ; 
    m_frame = new Frame(m_opticks) ; 
    m_interactor = new Interactor(m_hub) ; 

    m_interactor->setFrame(m_frame);
    m_interactor->setScene(m_scene);
    //m_interactor->setComposition(m_composition);
    
    m_scene->setInteractor(m_interactor);

    m_frame->setInteractor(m_interactor);
    m_frame->setComposition(m_composition);
    m_frame->setScene(m_scene);

    m_frame->setTitle(argv[0]);
    

    // App::prepareViz

    glm::uvec4 size = m_opticks->getSize();
    glm::uvec4 position = m_opticks->getPosition() ;

    m_composition->setSize( size );
    m_composition->setFramePosition( position );


    m_scene->setRenderMode("global"); 

    m_scene->initRenderers(); 
    m_frame->init();  
    m_window = m_frame->getWindow();
   
    //m_scene->setComposition(m_composition);     // defer until renderers are setup 
    m_scene->hookupRenderers();     // defer until renderers are setup 


    // App::uploadGeometryViz  (has to be after setting up the renderers)

    m_ggeo->setComposition(m_composition);

    m_scene->setGeometry(m_geolib);
    m_scene->uploadGeometry();

    bool autocam = true ;
    unsigned int target = 0 ; 
    m_scene->setTarget(target, autocam);


    // App::renderLoop

    m_frame->hintVisible(true);
    m_frame->show();
    LOG(info) << "after frame.show() "; 

    unsigned int count ; 

    while (!glfwWindowShouldClose(m_window))
    {    
        m_frame->listen(); 

        count = m_composition->tick();

        if( m_composition->hasChanged() || m_interactor->hasChanged() || count == 1)   
        {    
            render();

            glfwSwapBuffers(m_window);

            m_interactor->setChanged(false);  
            m_composition->setChanged(false);   // sets camera, view, trackball dirty status 
        }    
    }    
    return 0 ; 
}

