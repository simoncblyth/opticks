#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <string>
#include <vector>

class GUI {
  public:
       GUI();
       virtual ~GUI();

       void init(GLFWwindow* window);
       void newframe();
       void demo();
       void render();
       void shutdown();

       void choose( std::vector<std::pair<int, std::string> >& choices, bool* selection );
       //void choose( std::vector<std::pair<int, std::string> >& choices, std::vector<bool>& selection );
       //void choose( std::vector<std::pair<int, std::string> >& choices, std::vector<int>& selection );

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




