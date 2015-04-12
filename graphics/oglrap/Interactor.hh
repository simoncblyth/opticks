#pragma once

#include <vector>

class Camera ; 
class View ; 
class Trackball ;


// maybe this belongs in oglrap- 

class Interactor {
  public:
       static const char* DRAGFACTOR ; 
       static const char* OPTIXMODE ; 

       Interactor(); 
       void setup(Camera* camera, View* view, Trackball* trackball);

  public:
       bool isOptiXMode(){ return m_optix_mode > 0 ; }
       void setOptiXMode(int optix_mode){ m_optix_mode =  optix_mode ; }


  public:
       void cursor_drag( float x, float y, float dx, float dy );
       void key_pressed(unsigned int key);
       void key_released(unsigned int key);
       void Print(const char* msg);

       void configureF(const char* name, std::vector<float> values);
       void configureI(const char* name, std::vector<int> values);

       

  private:
       Camera*      m_camera ; 
       View*        m_view ; 
       Trackball*   m_trackball ; 

       bool m_zoom_mode ;
       bool m_pan_mode ;
       bool m_near_mode ;
       bool m_far_mode ;
       bool m_yfov_mode ;
       bool m_rotate_mode ;
       int  m_optix_mode ;

       float m_dragfactor ;

};


