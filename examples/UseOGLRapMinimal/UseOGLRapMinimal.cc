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

// oglrap/tests/AxisAppCheck.cc
//#include "AxisApp.hh"
//#include "Opticks.hh"
#include "OPTICKS_LOG.hh"


#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "Opticks.hh"
#include "Frame.hh"
#include "Interactor.hh"
#include "Composition.hh"
#include "Device.hh"

#include "NPY.hpp"
#include "Rdr.hh"

#include "OKConf.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);
    LOG(info) << argv[0] ; 

    Opticks ok(argc, argv, "--renderlooplimit 1000"); 
    ok.configure(); 


    Composition* m_composition = new Composition(&ok) ; 

    // OpticksViz::init
    Frame* m_frame = new Frame ; 
    m_frame->setComposition(m_composition);

    Interactor* m_interactor  = new Interactor(m_composition) ;
    m_interactor->setFrame(m_frame);
    m_frame->setInteractor(m_interactor);

    // AxisApp::init
    // Scene::initRenderers
    const char* m_shader_dir = OKConf::ShaderDir();    
    const char* m_shader_incl_path = OKConf::ShaderDir();    

    Device* m_device = new Device();
    Rdr* m_axis_renderer = new Rdr(m_device, "axis", m_shader_dir, m_shader_incl_path );

    //  OpticksViz::prepareScene
    m_frame->init(); 

    // Scene::hookupRenderers
    m_axis_renderer->setComposition( m_composition ) ; 

    glm::vec4 ce(0,0,0, 1000.); 
    bool autocam = true ; 
    m_composition->setCenterExtent( ce, autocam );  
    m_composition->update();

    // Scene::uploadAxis
    bool dbg = true ; 
    m_axis_renderer->upload(m_composition->getAxisAttr(), dbg);

    m_frame->hintVisible(true);
    m_frame->show();

    GLFWwindow* window = m_frame->getWindow();


    int count(0) ; 
    bool exitloop(false); 
    int renderlooplimit = ok.getRenderLoopLimit();

    while (!glfwWindowShouldClose(window) && !exitloop)
    {   
        m_frame->listen();
        m_frame->viewport();
        m_frame->clear();
        m_composition->update();
        if(count == 0 ) m_composition->Details("Details"); 

        m_axis_renderer->render();
        glfwSwapBuffers(window);
        count++ ; 

         exitloop = renderlooplimit > 0 && count > renderlooplimit ; 
    }   

    m_frame->exit();  //  

    //AxisApp aa(&ok); 
    //aa.renderLoop();
    return 0 ; 
}

/**
On mac this succeeds to pop up a window with an off-centered (why?) RGB axis 
On Linux : the issue manifests : only the blue line appears

**/



