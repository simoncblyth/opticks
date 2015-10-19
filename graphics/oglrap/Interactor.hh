#pragma once

#include <vector>
#include <string>
#include <cstring>
#include <cstdio>

class Composition ;
class Bookmarks ; 

class Camera ; 
class View ; 
class Trackball ;
class Clipper ;
class Touchable ; 
class Frame ;   
class Scene ;   


class Interactor {
  public:
       enum { NUM_KEYS = 512 } ;
       enum { 
              e_shift   = 1 << 0, 
              e_control = 1 << 1, 
              e_option  = 1 << 2, 
              e_command = 1 << 3 
            } ;
       unsigned int getModifiers();
       static std::string describeModifiers(unsigned int modifiers);

       static const char* keys ; 

       static const char* DRAGFACTOR ; 
       static const char* OPTIXMODE ; 
       static const char* GUIMODE ; 

       Interactor(); 
       void gui();
       void setComposition(Composition* composition);
       void setBookmarks(Bookmarks* bookmarks);
       void setTouchable(Touchable* touchable);
       void setScene(Scene* scene);
       void setFrame(Frame* frame);
       void setContainer(unsigned int container);

  public:
       bool isOptiXMode();
       void setOptiXMode(int optix_mode);
       int  getOptiXMode();
       void setOptiXResolutionScale(unsigned int scale);
       void nextOptiXResolutionScale(unsigned int modifiers);
       unsigned int getOptiXResolutionScale();

       Touchable*   getTouchable();
       Frame*       getFrame();
       bool*        getGuiModeAddress();
       bool*        getModeAddress(const char* name);
       unsigned int getContainer();

  public:
       void cursor_drag( float x, float y, float dx, float dy, int ix, int iy );

  public:
       void number_key_pressed(unsigned int number);
       void number_key_released(unsigned int number);
       void key_pressed(unsigned int key);
       void key_released(unsigned int key);

  public:
       void configureF(const char* name, std::vector<float> values);
       void configureI(const char* name, std::vector<int> values);

  public:
       void Print(const char* msg);
       void updateStatus();
       const char* getStatus(); 

  private:
       Composition* m_composition ; 
       Bookmarks*   m_bookmarks ; 
       Camera*      m_camera ; 
       View*        m_view ; 
       Trackball*   m_trackball ; 
       Clipper*     m_clipper ; 
       Touchable*   m_touchable ; 
       Frame*       m_frame; 
       Scene*       m_scene; 

       bool m_zoom_mode ;
       bool m_pan_mode ;
       static const unsigned int _pan_mode_key ;
       bool m_near_mode ;
       bool m_far_mode ;
       bool m_yfov_mode ;
       bool m_rotate_mode ;
       bool m_bookmark_mode ;
       bool m_gui_mode ;
       bool m_keys_down[NUM_KEYS] ; 

       int  m_optix_mode ;
       unsigned int m_optix_resolution_scale ;

       float m_dragfactor ;
       unsigned int m_container ;

       enum { STATUS_SIZE = 128 };
       char m_status[STATUS_SIZE] ; 

};

inline bool* Interactor::getGuiModeAddress()
{
    return getModeAddress(GUIMODE);
}

inline bool* Interactor::getModeAddress(const char* name)
{
    if(strcmp(name, GUIMODE)==0) return &m_gui_mode ;
    return NULL ;
}

inline Interactor::Interactor() 
   :
   m_composition(NULL),
   m_bookmarks(NULL),
   m_camera(NULL),
   m_view(NULL),
   m_trackball(NULL),
   m_clipper(NULL),
   m_touchable(NULL),
   m_frame(NULL),
   m_scene(NULL),
   m_zoom_mode(false), 
   m_pan_mode(false), 
   m_near_mode(false), 
   m_far_mode(false), 
   m_yfov_mode(false),
   m_rotate_mode(false),
   m_bookmark_mode(false),
   m_gui_mode(false),
   m_optix_mode(0),
   m_optix_resolution_scale(1),
   m_dragfactor(1.f),
   m_container(0)
{
   for(unsigned int i=0 ; i < NUM_KEYS ; i++) m_keys_down[i] = false ; 
   m_status[0] = '\0' ;
}



inline void Interactor::setScene(Scene* scene)
{
    m_scene = scene ; 
}
inline void Interactor::setTouchable(Touchable* touchable)
{
    m_touchable = touchable ; 
}
inline Touchable* Interactor::getTouchable()
{
    return m_touchable ; 
}
inline void Interactor::setFrame(Frame* frame)
{
    m_frame = frame ; 
}
inline Frame* Interactor::getFrame()
{
    return m_frame ; 
}


inline void Interactor::setContainer(unsigned int container)
{
    printf("Interactor::setContainer %u \n", container);
    m_container = container ; 
}

inline unsigned int Interactor::getContainer()
{
    return m_container ;  
}



inline bool Interactor::isOptiXMode()
{ 
    return m_optix_mode > 0 ; 
}
inline void Interactor::setOptiXMode(int optix_mode)
{
    m_optix_mode =  optix_mode ; 
}
inline int Interactor::getOptiXMode()
{ 
    return m_optix_mode ; 
}
inline unsigned int Interactor::getOptiXResolutionScale()
{ 
    return m_optix_resolution_scale  ; 
}
inline void Interactor::setOptiXResolutionScale(unsigned int scale)
{
    m_optix_resolution_scale = (scale > 0 && scale <= 32) ? scale : 1 ;  
    printf("Interactor::setOptiXResolutionScale %u \n", m_optix_resolution_scale);
}
inline void Interactor::nextOptiXResolutionScale(unsigned int modifiers)
{
    if(modifiers & e_shift)
        setOptiXResolutionScale(getOptiXResolutionScale()/2);
    else
        setOptiXResolutionScale(getOptiXResolutionScale()*2);
}
