#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <string>
#include <vector>

#include "PhotonsNPY.hpp"


class GUI {
  public:
       typedef std::vector<std::pair<int, std::string> > Choices_t ;

       GUI();
       virtual ~GUI();

       void init(GLFWwindow* window);
       void newframe();
       void demo();
       void render();
       void shutdown();

       void setupBoundarySelection(Choices_t* choices, bool* selection);

  private:
       void choose( std::vector<std::pair<int, std::string> >& choices, bool* selection );
       //void choose( std::vector<std::pair<int, std::string> >& choices, std::vector<bool>& selection );
       //void choose( std::vector<std::pair<int, std::string> >& choices, std::vector<int>& selection );

  private:
       bool  m_show_test_window ;
       bool  m_show_another_window ;
       float m_f ; 

       std::vector<std::pair<int, std::string> >* m_boundary_choices ;
       bool*                                      m_boundary_selection ;

};


inline GUI::GUI() 
   :
   m_show_test_window(true),
   m_show_another_window(false),
   m_f(0.0f),
   m_boundary_choices(NULL),
   m_boundary_selection(NULL)
{
}

inline void GUI::setupBoundarySelection(Choices_t* choices, bool* selection)
{
    m_boundary_choices = choices ; 
    m_boundary_selection = selection ; 
}


