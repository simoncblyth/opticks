#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <string>
#include <vector>

class OpticksHub ; 
class GItemIndex ; 
class OpticksAttrSeq ; 

class Interactor ; 
class Scene ; 
class Composition ;

class View ;
class TrackView ; 
class InterpolatedView ; 
class OrbitalView ; 

class Camera ;
class Clipper ;
class Trackball ;
class Bookmarks ;
class StateGUI ; 
class Photons ; 
class Animator ; 

#include "OGLRAP_API_EXPORT.hh"
#include "OGLRAP_HEAD.hh"

/**
GUI
====

Canonical m_gui instance is resident of OpticksViz that 
is instanciated by OpticksViz::prepareGUI just before the renderloop::

    132 void OpticksViz::visualize()
    133 {
    134     prepareGUI();
    135     renderLoop();
    136 }



**/


class OGLRAP_API GUI {
  public:
       // TODO: perhaps simplify to (unsigned int nlabel, const char**) 
       //       seems ImGui doesnt need to known the underly code
       //
       typedef std::vector<std::pair<int, std::string> > Choices_t ;

       GUI(OpticksHub* hub);
       virtual ~GUI();

       void setScene(Scene* scene);
       void setComposition(Composition* composition);
       void setInteractor(Interactor* interactor);
       void setPhotons(Photons* photons);
       void setView(View* view);
       void setCamera(Camera* camera);
       void setClipper(Clipper* clipper);
       void setTrackball(Trackball* trackball);
       void setBookmarks(Bookmarks* bookmarks);
       void setAnimator(Animator* animator);
       void setStateGUI(StateGUI* state_gui);

       void init(GLFWwindow* window);
       void newframe();
       void show(bool* opened);
       void show_scrubber(bool* opened);
       void show_label(bool* opened);
       void render();
       void shutdown();

       void setupHelpText(const std::string& txt);
       void setupStats(const std::vector<std::string>& stats);
       void setupParams(const std::vector<std::string>& params);


   public:
       void viewgui();
       void standard_view(View* view);
       void track_view(TrackView* tv);
       void orbital_view(OrbitalView* ov);
       void interpolated_view(InterpolatedView* iv);
       bool animator_gui(Animator* animator, const char* label, const char* fmt, float power);

       void camera_gui(Camera* camera);
       void trackball_gui(Trackball* trackball);
       void clipper_gui(Clipper* clipper);
       void bookmarks_gui(Bookmarks* bookmarks);
       void composition_gui(Composition* composition);

   public:
       //static void gui_item_index(GItemIndex* ii);
       static void gui_item_index(OpticksAttrSeq* al);
       static void gui_item_index(const char* type, std::vector<std::string>& labels, std::vector<unsigned int>& codes);


       static void gui_radio_select(GItemIndex* ii);

  private:
       void choose( unsigned int n, const char** choices, bool** selection );
  private:
       void choose( std::vector<std::pair<int, std::string> >* choices, bool* selection );
       //void choose( std::vector<std::pair<int, std::string> >& choices, std::vector<bool>& selection );
       //void choose( std::vector<std::pair<int, std::string> >& choices, std::vector<int>& selection );

  private:
       OpticksHub*   m_hub ; 
       bool          m_show_test_window ;
       float         m_bg_alpha ; 
       float         m_scrub_alpha ; 
       float         m_label_alpha ; 
       Interactor*   m_interactor ; 
       Scene*        m_scene ; 
       Composition*  m_composition ; 
       View*         m_view; 
       Camera*       m_camera; 
       Clipper*      m_clipper ; 
       Trackball*    m_trackball ; 
       Bookmarks*    m_bookmarks ; 
       StateGUI*     m_state_gui ; 
       Photons*      m_photons ; 
       Animator*     m_animator ; 

       std::string   m_help ; 
       std::vector<std::string> m_stats ; 
       std::vector<std::string> m_params ; 


};
#include "OGLRAP_TAIL.hh"


