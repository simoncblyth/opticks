#pragma once

#include <vector>
#include <string>

class Composition ;
class Bookmarks ; 

class Camera ; 
class View ; 
class Trackball ;
class Clipper ;
class Touchable ; 
class Frame ;   


class Interactor {
  public:
       static const char* DRAGFACTOR ; 
       static const char* OPTIXMODE ; 

       Interactor(); 
       void setComposition(Composition* composition);
       void setBookmarks(Bookmarks* bookmarks);
       void setTouchable(Touchable* touchable);
       void setFrame(Frame* frame);
       void setContainer(unsigned int container);

  public:
       bool isOptiXMode();
       void setOptiXMode(int optix_mode);
       int  getOptiXMode();

       Touchable* getTouchable();
       Frame*     getFrame();


  public:
       void cursor_drag( float x, float y, float dx, float dy );

  public:
       void number_key_pressed(unsigned int number, int ix, int iy);
       void number_key_released(unsigned int number, int ix, int iy);
       void key_pressed(unsigned int key, int ix, int iy );
       void key_released(unsigned int key, int ix, int iy );

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

       bool m_zoom_mode ;
       bool m_pan_mode ;
       bool m_near_mode ;
       bool m_far_mode ;
       bool m_yfov_mode ;
       bool m_rotate_mode ;
       bool m_jump_mode ;
       bool m_bookmark_mode ;

       int  m_optix_mode ;

       float m_dragfactor ;
       unsigned int m_container ;

       std::string m_status ; 

};


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
   m_zoom_mode(false), 
   m_pan_mode(false), 
   m_near_mode(false), 
   m_far_mode(false), 
   m_yfov_mode(false),
   m_rotate_mode(false),
   m_jump_mode(false),
   m_bookmark_mode(false),
   m_optix_mode(0),
   m_dragfactor(1.f),
   m_container(0)
{
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
    m_container = container ; 
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

 



