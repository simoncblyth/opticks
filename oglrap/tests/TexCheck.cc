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

/*
Search for a .ppm image file on your system and pass to FrameTest 

  SHADER_DIR=$(oglrap-sdir)/gl FrameTest /opt/local/lib/tk8.6/demos/images/teapot.ppm

Use oglrap-frametest


2017/8/18 
   runs, but not popping up the window

   * was missing the show 

*/

#include "NGLM.hpp"

#include "Opticks.hh"
#include "OpticksHub.hh"

#include "Frame.hh"
#include "Composition.hh"
#include "Renderer.hh"
#include "Interactor.hh"
#include "Texture.hh"
#include "G.hh"


#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);


    if(argc < 2)  
    {   
        printf("%s : expecting argument with path to ppm file\n", argv[0]);
        return 0 ;  
    }   
    char* ppmpath = argv[1] ;

    LOG(info) << argv[0] << " ppmpath " << ppmpath ; 

    Opticks ok(argc, argv);
    OpticksHub hub(&ok);


    G::VERBOSE = true ; 

    Frame frame ; 
    Composition composition(&ok) ; 
    //Interactor interactor(&hub) ;  // why the interactor needs the hub ?
    Interactor interactor(&composition) ; 
    Renderer renderer("tex") ; 

    // canonical wiring in OpticksViz::init
    interactor.setFrame(&frame);
    frame.setInteractor(&interactor);    // GLFW key and mouse events from frame to interactor
    //interactor.setComposition(&composition); // interactor changes camera, view, trackball 
    renderer.setComposition(&composition);  // composition provides matrices to renderer 

    Texture texture ;
    texture.loadPPM(ppmpath);
    composition.setSize(texture.getWidth(), texture.getHeight(),2 );

    frame.setComposition(&composition);
    frame.setTitle("TexTest");
    frame.init(); // creates OpenGL context 

    frame.hintVisible(true); 
    frame.show();

    //composition.setModelToWorld(texture.getModelToWorldPtr(0));   // point at the geometry 

    gfloat4 ce = texture.getCenterExtent(0);

    glm::vec4 ce_(ce.x, ce.y, ce.z, ce.w);

    LOG(info) << " ce (" << ce.x  << " " << ce.y << " " << ce.z << " " << ce.w << ")" ; 

    bool autocam = true ; 
    composition.setCenterExtent(ce_, autocam); // point at the geometry 

    composition.update();
    composition.Details("Composition::details");

    texture.create();   // after OpenGL context creation, done in frame.gl_init_window
    renderer.upload(&texture);

    GLFWwindow* window = frame.getWindow();

    while (!glfwWindowShouldClose(window))
    {
        frame.listen();
        frame.viewport();
        frame.clear();
        composition.update();

        renderer.render();
        glfwSwapBuffers(window);
    }

    frame.exit();  //

    return 0 ;
}


/**

The above code is missing something (that is done in standard running) that prevents linking of shaders.
First guess would be tex setup::

    [blyth@localhost oglrap]$ TexCheck /tmp/pix.ppm 
    2019-04-15 10:12:16.073 INFO  [87193] [main@43] TexCheck ppmpath /tmp/pix.ppm
    ...
    2019-04-15 10:23:20.080 FATAL [104897] [G::ErrCheck@57] Prog::create.m_id
    2019-04-15 10:23:20.080 FATAL [104897] [G::ErrCheck@57] Prog::create.]
    ERROR: linking GL shader program index 1 
    2019-04-15 10:23:20.080 INFO  [104897] [Prog::_print_program_info_log@275] Prog::_print_program_info_log Prog  tag:tex verbosity:0
    2019-04-15 10:23:20.080 INFO  [104897] [ProgLog::dump@21] Prog::_print_program_info_log
    ProgLog::dump id 1:
    []2019-04-15 10:23:20.080 INFO  [104897] [Prog::_print_program_info_log@280]  NO_FRAGMENT_SHADER 0
    Prog::link ERROR
    GL_LINK_STATUS = 0
    GL_ATTACHED_SHADERS = 0
    GL_ACTIVE_ATTRIBUTES = 0
    GL_ACTIVE_UNIFORMS = 0
    GL_VALIDATE_STATUS = 0
    [blyth@localhost tests]$ 


**/


