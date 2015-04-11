#ifndef FRAME_H
#define FRAME_H

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "gleq.h"

#include <string>
#include <vector>

class Config ;
class Scene ;
class Interactor ; 


class Frame {
   public:
       Frame();
       virtual ~Frame();
       
       void configureI(const char* name, std::vector<int> values);
       void configureS(const char* name, std::vector<std::string> values);
       void setSize(unsigned int width, unsigned int height, unsigned int coord2pixel=2);
       void setTitle(const char* title);
       void setScene(Scene* scene);
       void setInteractor(Interactor* interactor);
       void setDumpevent(int dumpevent);

       void init_window();
       void exit();
    
       void listen();
       void render();
       GLFWwindow* getWindow(){ return m_window ; }

   public:
       unsigned int getWidth(){  return m_width ; } 
       unsigned int getHeight(){ return m_height ; } 
       unsigned int getCoord2pixel(){ return m_coord2pixel ; } 

   private:
       void handle_event(GLEQevent& event);
       void dump_event(GLEQevent& event);
       void resize(unsigned int width, unsigned int height);

   private:
       void key_pressed(unsigned int key);
       void key_released(unsigned int key);

   private:
       unsigned int  m_width ; 
       unsigned int  m_height ; 
       unsigned int  m_coord2pixel ; 
       const char*   m_title ;
       GLFWwindow*   m_window;
       Scene*        m_scene ;
       Interactor*   m_interactor ; 
       bool          m_cursor_inwindow ; 
       float         m_cursor_x ; 
       float         m_cursor_y ; 
       unsigned int  m_dumpevent ; 


};

#endif


