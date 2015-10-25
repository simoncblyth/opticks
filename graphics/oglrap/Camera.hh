#pragma once

#include <glm/glm.hpp>  
#include <math.h> 
#include "stdio.h"
#include <vector>
#include <string>
#include "Configurable.hh"

class Camera : public Configurable  {
  public:

     static const char* PRINT ; 

     static const char* NEAR ; 
     static const char* FAR ; 
     static const char* ZOOM ; 
     static const char* SCALE ; 

     static const char* PARALLEL ; 

     Camera(int width=1024, int height=768, float near=0.1f, float far=10000.f, float zoom=1.f, float scale=35.f,  bool parallel=false) ;

     glm::mat4 getProjection();
     glm::mat4 getPerspective();
     glm::mat4 getOrtho();
     glm::mat4 getFrustum();

     bool hasChanged();
     void setChanged(bool changed); 

   public:
       typedef enum { PERSPECTIVE_CAMERA, OTHOGRAPHIC_CAMERA, NUM_CAMERA_STYLE } Style_t ;
       void nextStyle();
       void setStyle(Camera::Style_t style);
       Camera::Style_t getStyle();

 public:
     // Configurable
     std::vector<std::string> getTags();
     void set(const char* name, std::string& xyz);
     std::string get(const char* name);

     void gui();

     static bool accepts(const char* name);

     // NB dont overload these as it confuses boost::bind
     void configureF(const char* name, std::vector<float> values);
     void configureI(const char* name, std::vector<int> values);
     void configureS(const char* name, std::vector<std::string> values);

     void configure(const char* name, const char* value);
     void configure(const char* name, float value);

  public:
     // mouse control
     void near_to( float x, float y, float dx, float dy );
     void far_to( float x, float y, float dx, float dy );
     void zoom_to( float x, float y, float dx, float dy );
     void scale_to( float x, float y, float dx, float dy );
  public:
     // infrequent inputs 
     void setParallel(bool parallel);
     void setSize(int width, int height );
     void setPixelFactor(unsigned int factor);
  public:
     unsigned int getWidth();
     unsigned int getHeight();
     unsigned int getPixelWidth();
     unsigned int getPixelHeight();
     unsigned int getPixelFactor();
  public:
     void setNearClip(float _min, float _max);
     void setFarClip(float _min, float _max);
     void setZoomClip(float _min, float _max);
     void setScaleClip(float _min, float _max);
  public:
     bool getParallel();
     float getAspect(); // width/height (> 1 for landscape)
  public:
     // interactive inputs
     void setNear(float near);
     void setFar(float far);
     void setZoom(float zoom);
     void setScale(float scale);
  public:
     void setYfov(float yfov_degrees);  // alternative way to set zoom (= 1./tan(yfov/2))
  public:
     float getNear();
     float getFar();
     float getZoom();
     float getScale();
  public:
     float getDepth();
  public:
     float getYfov();
     float getTanYfov();
  public:
     // the below top/bottom/left/right formerly scaled with  m_near,  now scaling with m_scale
     float getTop();
     float getBottom();
     float getLeft();
     float getRight();
  public:
     void Print(const char* msg="Camera::Print");
     void Summary(const char* msg="Camera::Summary");

   private:
     int   m_size[2] ;
     int   m_pixel_factor ; 
     float m_aspect ; 

     float m_nearclip[2] ;
     float m_farclip[2] ;
     float m_zoomclip[2] ;
     float m_scaleclip[2] ;

     float m_near ;
     float m_far ;
     float m_zoom ;
     float m_scale ; 
     float m_ortho_kludge ; 

     bool m_parallel ; 
     bool m_changed ; 

};



inline Camera::Camera(int width, int height, float near, float far, float zoom, float scale, bool parallel) 
       :
         m_changed(true),
         m_ortho_kludge(8.0)
{
    setSize(width, height);
    setPixelFactor(1); 

    setNearClip(1e-6, 1e6);
    setFarClip(1e-6, 1e6);
    setZoomClip(0.01f, 100.f);
    setScaleClip(1.f, 10000.f);

    setNear(near);
    setFar(far);
    setZoom(zoom);
    setScale(scale);

    setParallel(parallel);
} 


inline bool Camera::hasChanged()
{
    return m_changed ; 
}
inline void Camera::setChanged(bool changed)
{
    m_changed = changed ; 
}


inline void Camera::nextStyle()
{
    int next = (getStyle() + 1) % NUM_CAMERA_STYLE ; 
    setStyle( (Style_t)next ) ; 
}
inline void Camera::setStyle(Style_t style)
{
    m_parallel = int(style) == 1  ;
}
inline Camera::Style_t Camera::getStyle()
{
    return (Style_t)(m_parallel ? 1 : 0 ) ;
}






inline void Camera::setParallel(bool parallel)
{
    m_parallel = parallel ;
    m_changed = true ; 
}
inline bool Camera::getParallel(){ return m_parallel ; }

inline void Camera::setSize(int width, int height )
{
    m_size[0] = width ;
    m_size[1] = height ;
    m_aspect  = (float)width/(float)height ;   // (> 1 for landscape) 
    m_changed = true ; 
}
inline void Camera::setPixelFactor(unsigned int factor)
{
    m_pixel_factor = factor ; 
    m_changed = true ; 
}

inline unsigned int Camera::getWidth(){  return m_size[0]; }
inline unsigned int Camera::getHeight(){ return m_size[1]; }
inline float        Camera::getAspect(){ return m_aspect ; }

inline unsigned int Camera::getPixelWidth(){  return m_size[0]*m_pixel_factor; }
inline unsigned int Camera::getPixelHeight(){ return m_size[1]*m_pixel_factor; }
inline unsigned int Camera::getPixelFactor(){ return m_pixel_factor ; }




inline void Camera::near_to( float x, float y, float dx, float dy )
{
    setNear(m_near + m_near*dy );
    //printf("Camera::near_to %10.3f \n", m_near);
}
inline void Camera::far_to( float x, float y, float dx, float dy )
{
    setFar(m_far + m_far*dy );
    //printf("Camera::far_to %10.3f \n", m_far);
}
inline void Camera::zoom_to( float x, float y, float dx, float dy )
{
    setZoom(m_zoom + 30.*dy) ;
    //printf("Camera::zoom_to %10.3f \n", m_zoom);
}
inline void Camera::scale_to( float x, float y, float dx, float dy )
{
    setScale(m_scale + 30.*dy) ;
    //printf("Camera::scale_to %10.3f \n", m_scale);
}




inline void Camera::setNear(float near)
{
    if(      near < m_nearclip[0] )  m_near = m_nearclip[0] ;
    else if( near > m_nearclip[1] )  m_near = m_nearclip[1] ;
    else                             m_near = near ;
    m_changed = true ; 
}
inline void Camera::setFar(float far)
{
    if(      far < m_farclip[0] )  m_far = m_farclip[0] ;
    else if( far > m_farclip[1] )  m_far = m_farclip[1] ;
    else                           m_far = far ;
    m_changed = true ; 
}
inline void Camera::setZoom(float zoom)
{
    if(      zoom < m_zoomclip[0] )  m_zoom = m_zoomclip[0] ;
    else if( zoom > m_zoomclip[1] )  m_zoom = m_zoomclip[1] ;
    else                             m_zoom = zoom ;
    m_changed = true ; 
}
inline void Camera::setScale(float scale)
{
    if(      scale < m_scaleclip[0] )  m_scale = m_scaleclip[0] ;
    else if( scale > m_scaleclip[1] )  m_scale = m_scaleclip[1] ;
    else                               m_scale = scale ;
    m_changed = true ; 
}



inline float Camera::getNear(){  return m_near ; }
inline float Camera::getFar(){   return m_far ;  }
inline float Camera::getZoom(){  return m_zoom ; } 
inline float Camera::getScale(){ return m_parallel ? m_scale*m_ortho_kludge : m_scale ; }

inline float Camera::getDepth(){   return m_far - m_near ; }
inline float Camera::getTanYfov(){ return 1.f/m_zoom ; }  // actually tan(Yfov/2)

inline float Camera::getTop(){    return getScale() / m_zoom ; }
inline float Camera::getBottom(){ return -getScale() / m_zoom ; }
inline float Camera::getLeft(){   return -m_aspect * getScale() / m_zoom ; } 
inline float Camera::getRight(){  return  m_aspect * getScale() / m_zoom ; } 

/*
zoom defined as apparent size of an object relative to the size for a 90 degree field of view

   tan( 90/2 ) = 1.

   math.tan(90.*0.5*math.pi/180.)
   0.9999999999999999

              + . . . +
             /|       *
            / |       *
           /  |       *
          /   |       * 
         /    +       *
        /  .  *       *
       / .    *       *
     -+-------*-------*------    
       \ .    *       *
        \  .  *       *
         \    +       *
          \  zoom~1   *
           \  |       *
            \ |       *
             \|       *
              + . . . +
                    zoom~2

*/

inline void Camera::setYfov(float yfov)
{
    // setYfov(90.) -> setZoom(1.)

    // fov = 2atan(1/zoom)
    // zoom = 1/tan(fov/2)

    float zoom = 1.f/tan(yfov*0.5f*M_PI/180.f );
    setZoom( zoom );
}
inline float Camera::getYfov()
{
    return 2.f*atan(1./m_zoom)*180./M_PI;
}


inline void Camera::setNearClip(float _min, float _max)
{
    m_nearclip[0] = _min ;  
    m_nearclip[1] = _max ;  
}
inline void Camera::setFarClip(float _min, float _max)
{
    m_farclip[0] = _min ;  
    m_farclip[1] = _max ;  
}
inline void Camera::setZoomClip(float _min, float _max)
{
    m_zoomclip[0] = _min ;  
    m_zoomclip[1] = _max ;  
}
inline void Camera::setScaleClip(float _min, float _max)
{
    m_scaleclip[0] = _min ;  
    m_scaleclip[1] = _max ;  
}

