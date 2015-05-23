#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <string>
#include <vector>

class Scene ; 
class Composition ;
class View ;
class Camera ;
class Clipper ;
class Trackball ;

#include "PhotonsNPY.hpp"


/*
   Hmm controlling the layout of the GUI demands
   the setup and then do structure ?

   Unless scatter the ImGui code ?
*/

class GUI {
  public:
       // TODO: perhaps simplify to (unsigned int nlabel, const char**) 
       //       seems ImGui doesnt need to known the underly code
       //
       typedef std::vector<std::pair<int, std::string> > Choices_t ;

       GUI();
       virtual ~GUI();

       void setScene(Scene* scene);
       void setComposition(Composition* composition);
       void setView(View* view);
       void setCamera(Camera* camera);
       void setClipper(Clipper* clipper);
       void setTrackball(Trackball* trackball);

       void init(GLFWwindow* window);
       void newframe();
       void show(bool* opened);
       void render();
       void shutdown();

       void setupHelpText(std::string txt);
       void setupBoundarySelection(Choices_t* choices, bool* selection);


  private:
       void choose( unsigned int n, const char** choices, bool** selection );
  private:
       void choose( std::vector<std::pair<int, std::string> >* choices, bool* selection );
       //void choose( std::vector<std::pair<int, std::string> >& choices, std::vector<bool>& selection );
       //void choose( std::vector<std::pair<int, std::string> >& choices, std::vector<int>& selection );

  private:
       bool  m_show_test_window ;
       float       m_bg_alpha ; 
       std::string m_help ; 

       std::vector<std::pair<int, std::string> >* m_boundary_choices ;
       bool*                                      m_boundary_selection ;

       Scene*        m_scene ; 
       Composition*  m_composition ; 
       View*         m_view; 
       Camera*       m_camera; 
       Clipper*      m_clipper ; 
       Trackball*    m_trackball ; 

};


inline GUI::GUI() 
   :
   m_show_test_window(false),
   m_bg_alpha(0.65f),
   m_boundary_choices(NULL),
   m_boundary_selection(NULL),
   m_scene(NULL),
   m_composition(NULL),
   m_view(NULL),
   m_camera(NULL),
   m_trackball(NULL),
   m_clipper(NULL)
{
}


inline void GUI::setScene(Scene* scene)
{
    m_scene = scene ; 
}
inline void GUI::setView(View* view)
{
    m_view = view ; 
}
inline void GUI::setCamera(Camera* camera)
{
    m_camera = camera ; 
}
inline void GUI::setClipper(Clipper* clipper)
{
    m_clipper = clipper ; 
}
inline void GUI::setTrackball(Trackball* trackball)
{
    m_trackball = trackball ; 
}




inline void GUI::setupBoundarySelection(Choices_t* choices, bool* selection)
{
    m_boundary_choices = choices ; 
    m_boundary_selection = selection ; 
}


