#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

class GUI {
  public:
       GUI();
       virtual ~GUI();

       void init(GLFWwindow* window);
       void newframe();
       void demo();
       void render();
       void shutdown();

  private:
        bool  m_show_test_window ;
        bool  m_show_another_window ;
        float m_f ; 

};


inline GUI::GUI() 
   :
   m_show_test_window(true),
   m_show_another_window(false),
   m_f(0.0f)
{
}




