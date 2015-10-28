#pragma once

#include <glm/glm.hpp>  
#include <math.h> 
#include "stdio.h"
#include <vector>
#include <string>
#include "Configurable.hh"


/*

Perspective Projection
-------------------------

Matrix from getFrustum() uses::

        glm::frustum( getLeft(), getRight(), getBottom(), getTop(), getNear(), getFar() )

with inputs:

          left          -m_aspect*getScale()/m_zoom
          right          m_aspect*getScale()/m_zoom 
          top            getScale()/m_zoom
          bottom        -getScale()/m_zoom
          near 
          far

Note the opposing getScale() and zoom

The matrix is of below form, note the -1 which preps the perspective divide by z 
on converting from homogenous (see glm-)
 
     |   2n/w    0         0         (r+l)/(r-l)     | 
     |    0     2n/h       0         (t+b)/(t-b)     |
     |    0      0    -(f+n)/(f-n)   -2 f n/(f-n)    |
     |    0      0         -1           0            |
 

Normally this results in the FOV changing on changing the near distance.  
To avoid this getScale() returns m_near for perspective projection.  
This means the effective screen size scales with m_near, so FOV stays 
constant as m_near is varied, the only effect of changing m_near is 
to change the clipping of objects.

In order to change the FOV use the zoom setting.
 
  
Orthographic Projection
------------------------

Matrix from getOrtho() uses the below with same inputs as perspective::

      glm::ortho( getLeft(), getRight(), getBottom(), getTop(), getNear(), getFar() );

has form (see glm-):

    | 2/w  0   0        -(r+l)/(r-l)   |
    |  0  2/h  0        -(t+b)/(t-b)   |
    |  0   0  -2/(f-n)  -(f+n)/(f-n)   |
    |  0   0   0          1            |


Camera::aim invoked by Composition::aim
-------------------------------------------

*aim* sets near/far/scale given a basis length

   float a_near = basis/10. ;
   float a_far  = basis*5. ;
   float a_scale = basis ; 

Using gazelength basis matches raygen in the ray tracer, 
so OptiX and OpenGL renders line up.


Zoom
-----

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


class Camera : public Configurable  {
  public:

     static const char* PRINT ; 

     static const char* NEAR ; 
     static const char* FAR ; 
     static const char* ZOOM ; 
     static const char* SCALE ; 

     static const char* PARALLEL ; 

     Camera(int width=1024, int height=768, float basis=1000.f ) ;

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
     // sets near, far and the ortho scale 
     void aim(float basis);
  public:
     // interactive inputs
     void setNear(float near);
     void setFar(float far);
     void setZoom(float zoom);
     void setScale(float scale);
  private:
     void setBasis(float basis);
     void setYfov(float yfov_degrees);  // alternative way to set zoom (= 1./tan(yfov/2))
  public:
     float getBasis();
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

     float m_basis ; 
     float m_near ;
     float m_far ;
     float m_zoom ;
     float m_scale ; 

     bool m_parallel ; 
     bool m_changed ; 

};



inline Camera::Camera(int width, int height, float basis ) 
       :
         m_zoom(1.0f),
         m_parallel(false),
         m_changed(true)
{
    setSize(width, height);
    setPixelFactor(1); 

    aim(basis);

    setZoomClip(0.01f, 100.f);
} 

inline void Camera::aim(float basis)
{
   float a_near = basis/10. ;
   float a_far  = basis*5. ;
   float a_scale = basis ; 

   //printf("Camera::aim basis %10.4f a_near %10.4f a_far %10.4f a_scale %10.4f \n", basis, a_near, a_far, a_scale );

   setBasis(basis);
   setNear( a_near );
   setFar(  a_far );

   setNearClip( a_near/10.,  a_near*10.) ;
   setFarClip(  a_far/10.,   a_far*10. );
   setScaleClip( a_scale/10., a_scale*10. );

   setScale( a_scale );  // scale should be renamed to Ortho scale, as only relevant to Orthographic projection
}

inline void Camera::setBasis(float basis)
{
    m_basis = basis ; 
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



inline float Camera::getBasis(){ return m_basis ; } 

inline float Camera::getNear(){  return m_near ; }
inline float Camera::getFar(){   return m_far ;  }
inline float Camera::getZoom(){  return m_zoom ; } 

inline float Camera::getScale(){ return m_parallel ? m_scale  : m_near ; }

inline float Camera::getDepth(){   return m_far - m_near ; }
inline float Camera::getTanYfov(){ return 1.f/m_zoom ; }  // actually tan(Yfov/2)

inline float Camera::getTop(){    return getScale() / m_zoom ; }
inline float Camera::getBottom(){ return -getScale() / m_zoom ; }
inline float Camera::getLeft(){   return -m_aspect * getScale() / m_zoom ; } 
inline float Camera::getRight(){  return  m_aspect * getScale() / m_zoom ; } 

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


