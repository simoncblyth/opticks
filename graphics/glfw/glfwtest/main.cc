#include <stdlib.h>  //exit()
#include <stdio.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#define GLEQ_IMPLEMENTATION
#include "gleq.h"

#include "Config.hh"
#include "app.hh"
#include "Scene.hh"

#include "numpydelegate.hh"
#include "numpyserver.hpp"


int main(int argc, char** argv)
{
    Config config;

    config.parse(argc,argv);

    if(config.isAbort()) exit(EXIT_SUCCESS);

    App app(&config) ;

    numpydelegate nde ; 
    numpyserver<numpydelegate> srv(&nde, config.getUdpPort(), config.getZMQBackend());

    app.setSize(640,480);
    app.setTitle("Demo");
    app.init();

    Scene scene ; 
    scene.load("GLFWTEST_") ;
    scene.init();

    app.setScene(&scene);

    GLFWwindow* window = app.getWindow();

    while (!glfwWindowShouldClose(window))
    {
        app.listen(); 

        // give numpyserver a few cycles, to complete posts from the net thread
        // resulting in the non-blocking handler methods of the delegate being called
        srv.poll_one();  

        app.render();
        glfwSwapBuffers(window);
    }
    srv.stop();


    app.exit();

    exit(EXIT_SUCCESS);
}

