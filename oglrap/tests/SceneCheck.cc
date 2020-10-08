/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

/**
SceneCheck
============

TODO: get this working again, currently exename_allowed asserts in BOpticksResource::setupViaKey

**/



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

#include "OPTICKS_LOG.hh"


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
    OPTICKS_LOG(argc, argv);
    LOG(info) << argv[0] ; 

    Opticks* m_ok = new Opticks(argc, argv);
    OpticksHub* m_hub = new OpticksHub(m_ok);

    //  hmm below is now done inside the Hub 
    m_ggeo = new GGeo(m_ok);
    m_ggeo->loadFromCache();
    m_ggeo->dumpStats();

    m_geolib = m_ggeo->getGeoLib();

    // see App::initViz

    m_composition = new Composition(m_ok) ; 

    m_scene = new Scene(m_hub) ; 
    m_frame = new Frame() ; 
    //m_interactor = new Interactor(m_hub) ; 
    m_interactor = new Interactor(m_composition) ; 

    m_interactor->setFrame(m_frame);
    m_interactor->setScene(m_scene);
    //m_interactor->setComposition(m_composition);
    
    m_scene->setInteractor(m_interactor);

    m_frame->setInteractor(m_interactor);
    m_frame->setComposition(m_composition);
    m_frame->setScene(m_scene);

    m_frame->setTitle(argv[0]);
    

    // App::prepareViz

    glm::uvec4 size = m_ok->getSize();
    glm::uvec4 position = m_ok->getPosition() ;

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

    //m_scene->setGeometry(m_geolib);
    m_scene->setGeometry(m_ggeo);

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

