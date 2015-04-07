#ifndef APP_H
#define APP_H

#include <string>
#include <vector>

class Config ;
class Scene ;

class App {
   public:
       App();
       virtual ~App();
       
       void configureS(const char* name, std::vector<std::string> values);
       void setSize(unsigned int width, unsigned int height, unsigned int coord2pixel=2);
       void setTitle(const char* title);
       void setScene(Scene* scene);

       void init();
       void exit();
    
       void listen();
       void render();
       GLFWwindow* getWindow(){ return m_window ; }

   private:
       void handle_event(GLEQevent& event);
       void dump_event(GLEQevent& event);
       void resize(unsigned int width, unsigned int height);
       void key_pressed(unsigned int key);

   private:
       unsigned int  m_width ; 
       unsigned int  m_height ; 
       unsigned int  m_coord2pixel ; 
       const char*   m_title ;
       GLFWwindow*   m_window;
       Scene*        m_scene ;


};

#endif


