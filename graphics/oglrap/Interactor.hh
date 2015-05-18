#pragma once

#include <vector>


class Composition ;
class Bookmarks ; 

class Camera ; 
class View ; 
class Trackball ;
class Clipper ;
class Touchable ; 
  


class Interactor {
  public:
       static const char* DRAGFACTOR ; 
       static const char* OPTIXMODE ; 

       Interactor(); 
       void setComposition(Composition* composition);
       void setBookmarks(Bookmarks* bookmarks);
       void setTouchable(Touchable* touchable);

  public:
       bool isOptiXMode();
       void setOptiXMode(int optix_mode);
       int  getOptiXMode();

       Touchable* getTouchable();


  public:
       void cursor_drag( float x, float y, float dx, float dy );
       void number_key_pressed(unsigned int number);
       void key_pressed(unsigned int key, int ix, int iy );
       void key_released(unsigned int key, int ix, int iy );
       void Print(const char* msg);

       void configureF(const char* name, std::vector<float> values);
       void configureI(const char* name, std::vector<int> values);


  private:
       Composition* m_composition ; 
       Bookmarks*   m_bookmarks ; 
       Camera*      m_camera ; 
       View*        m_view ; 
       Trackball*   m_trackball ; 
       Clipper*     m_clipper ; 
       Touchable*   m_touchable ; 

       bool m_zoom_mode ;
       bool m_pan_mode ;
       bool m_near_mode ;
       bool m_far_mode ;
       bool m_yfov_mode ;
       bool m_rotate_mode ;
       bool m_jump_mode ;

       int  m_optix_mode ;

       float m_dragfactor ;

};


inline void Interactor::setTouchable(Touchable* touchable)
{
    m_touchable = touchable ; 
}
inline Touchable* Interactor::getTouchable()
{
    return m_touchable ; 
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

 



