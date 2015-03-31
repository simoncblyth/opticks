#ifndef APP_H
#define APP_H

#include <boost/asio.hpp>
#include <boost/thread.hpp>

#include "udpServer.hh"
#include "EventQueue.hh"

class Config ;
class Scene ;

class App {
   public:
       App(Config* config);
       virtual ~App();
       
       void setSize(unsigned int width, unsigned int height);
       void setTitle(const char* title);
       void setScene(Scene* scene);

       void init();
       void runloop();
       void exit();

   private:
       void init_net();
       void init_graphics();

   private:
       void listen();
       void listenUDP();

   private:
       void handle_event(GLEQevent& event);
       void dump_event(GLEQevent& event);
       void resize(unsigned int width, unsigned int height);
       void key_pressed(unsigned int key);
       void render();

   private:
       unsigned int m_width ; 
       unsigned int m_height ; 
       const char* m_title ;
       GLFWwindow* m_window;
       Scene* m_scene ;

   private:
       Config*                 m_config ;
       boost::asio::io_service m_ioService;
       boost::thread*          m_netThread ; 
       udpServer               m_udpServer ; 
       EventQueue              m_eventQueue ;
       boost::mutex            m_eventQueueMutex ;



};

#endif


