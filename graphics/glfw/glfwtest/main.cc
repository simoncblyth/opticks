#include <stdlib.h>  //exit()
#include <stdio.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#define GLEQ_IMPLEMENTATION
#include "gleq.h"

#include "Config.hh"
#include "app.hh"
#include "Scene.hh"


#include "net_server.hpp"



class Receiver {
    public:
        Receiver(){}

        void on_msg(std::string msg)
        {
           std::cout << std::setw(20) << boost::this_thread::get_id()
                     << " Receiver::on_msg "
                     << " msg ["  << msg << "] "
                     << std::endl;
        }   

        void on_npy(std::vector<int> shape, std::vector<float> data, std::string metadata)
        {   
           std::cout << std::setw(20) << boost::this_thread::get_id()
              << " Receiver::on_npy "
              << " shape dimension " << shape.size()
              << " data size " << data.size()
              << " metadata [" << metadata << "]"
              << std::endl ;
        }
};



int main(int argc, char** argv)
{
    Config config;
    config.parse(argc,argv);
    if(config.isAbort()) exit(EXIT_SUCCESS);

    App app(&config) ;
    //Receiver rcv ; 

    net_server<App> srv(&app, config.getUdpPort(), config.getZMQBackend());

    app.setSize(640,480);
    app.setTitle("Demo");
    app.init();

    Scene scene ; 
    scene.load("GLFWTEST_") ;
    scene.init();
    app.setScene(&scene);

    //app.runloop();

    GLFWwindow* window = app.getWindow();

    while (!glfwWindowShouldClose(window))
    {
        app.listen(); 
        //listenUDP(); 
        srv.poll_one();  // give net_app a few cycles, to complete posts from the net thread
        app.render();

        glfwSwapBuffers(window);
    }
    srv.stop();

   /*
    for(unsigned int i=0 ; i < 20 ; ++i )
    {   
        srv.poll_one();
        srv.sleep(1);
    }   
    srv.stop();
    */


    app.exit();

    exit(EXIT_SUCCESS);
}

