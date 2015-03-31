#include <stdlib.h>  //exit()
#include <stdio.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#define GLEQ_IMPLEMENTATION
#include "gleq.h"

#include "Config.hh"
#include "app.hh"
#include "Scene.hh"

int main(void)
{
    Config config ;      
    config.setUdpPort(8080);

    App app(&config) ;
    app.setSize(640,480);
    app.setTitle("Demo");
    app.init();

    Scene scene ; 
    scene.load("GLFWTEST_") ;
    scene.init();

    app.setScene(&scene);

    app.runloop();
    app.exit();

    exit(EXIT_SUCCESS);
}

