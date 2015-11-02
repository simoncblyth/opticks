#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <string>
#include <vector>

class GGeo ; 
//class GLoader ; 
class GItemIndex ; 
class GAttrSeq ; 

class Interactor ; 
class Scene ; 
class Composition ;
class View ;
class Camera ;
class Clipper ;
class Trackball ;
class Bookmarks ;
class Photons ; 


class GUI {
  public:
       // TODO: perhaps simplify to (unsigned int nlabel, const char**) 
       //       seems ImGui doesnt need to known the underly code
       //
       typedef std::vector<std::pair<int, std::string> > Choices_t ;

       GUI(GGeo* ggeo);
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
       //void setLoader(GLoader* loader);  // used to access GItemIndex materials, surfaces, flags 

       void init(GLFWwindow* window);
       void newframe();
       void show(bool* opened);
       void render();
       void shutdown();

       void setupHelpText(const std::string& txt);
       void setupStats(const std::vector<std::string>& stats);
       void setupParams(const std::vector<std::string>& params);

   public:
       //static void gui_item_index(GItemIndex* ii);
       static void gui_item_index(GAttrSeq* al);
       static void gui_item_index(const char* type, std::vector<std::string>& labels, std::vector<unsigned int>& codes);


       static void gui_radio_select(GItemIndex* ii);

  private:
       void choose( unsigned int n, const char** choices, bool** selection );
  private:
       void choose( std::vector<std::pair<int, std::string> >* choices, bool* selection );
       //void choose( std::vector<std::pair<int, std::string> >& choices, std::vector<bool>& selection );
       //void choose( std::vector<std::pair<int, std::string> >& choices, std::vector<int>& selection );

  private:
       GGeo*         m_ggeo ; 
       bool          m_show_test_window ;
       float         m_bg_alpha ; 
       Interactor*   m_interactor ; 
       Scene*        m_scene ; 
       Composition*  m_composition ; 
       View*         m_view; 
       Camera*       m_camera; 
       Clipper*      m_clipper ; 
       Trackball*    m_trackball ; 
       Bookmarks*    m_bookmarks ; 
       Photons*      m_photons ; 
       //GLoader*      m_loader ; 

       std::string   m_help ; 
       std::vector<std::string> m_stats ; 
       std::vector<std::string> m_params ; 


};


inline GUI::GUI(GGeo* ggeo) 
   :
   m_ggeo(ggeo),
   m_show_test_window(false),
   m_bg_alpha(0.65f),
   m_interactor(NULL),
   m_scene(NULL),
   m_composition(NULL),
   m_view(NULL),
   m_camera(NULL),
   m_clipper(NULL),
   m_trackball(NULL),
   m_bookmarks(NULL),
   m_photons(NULL)
   //m_loader(NULL)
{
}

inline void GUI::setInteractor(Interactor* interactor)
{
    m_interactor = interactor ; 
}
inline void GUI::setPhotons(Photons* photons)
{
    m_photons = photons ; 
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
inline void GUI::setBookmarks(Bookmarks* bookmarks)
{
    m_bookmarks = bookmarks ; 
}

//inline void GUI::setLoader(GLoader* loader)
//{
//    m_loader = loader ; 
//}


inline void GUI::setupHelpText(const std::string& txt)
{
    m_help = txt ; 
} 

inline void GUI::setupStats(const std::vector<std::string>& stats)
{
    m_stats = stats ; 
}
inline void GUI::setupParams(const std::vector<std::string>& params)
{
    m_params = params ; 
}


