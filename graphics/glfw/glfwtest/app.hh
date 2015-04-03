#ifndef APP_H
#define APP_H

#include <string>
#include <vector>

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
       void exit();
    
       void listen();
       void render();
       GLFWwindow* getWindow(){ return m_window ; }

   public:
       // handlers arriving from the network thread via net_app and boost::asio magic 
       // so long as call app.poll_one() within the event loop
       void on_msg(std::string msg);
       void on_npy(std::vector<int> shape, std::vector<float> data, std::string metadata);

   private:
       void handle_event(GLEQevent& event);
       void dump_event(GLEQevent& event);
       void resize(unsigned int width, unsigned int height);
       void key_pressed(unsigned int key);

   private:
       Config*       m_config ;
       unsigned int  m_width ; 
       unsigned int  m_height ; 
       const char*   m_title ;
       GLFWwindow*   m_window;
       Scene*        m_scene ;


};

#endif


