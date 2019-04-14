// oglrap/tests/AxisAppCheck.cc
//#include "AxisApp.hh"
//#include "Opticks.hh"
#include "OPTICKS_LOG.hh"


#include <GL/glew.h>
#include <GLFW/glfw3.h>

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

    Composition* m_composition = new Composition ; 

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

    while (!glfwWindowShouldClose(window))
    {   
        m_frame->listen();
        m_frame->viewport();
        m_frame->clear();
        m_composition->update();
        if(count == 0 ) m_composition->Details("Details"); 

        m_axis_renderer->render();
        glfwSwapBuffers(window);
        count++ ; 
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



