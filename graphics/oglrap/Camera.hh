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
     static const char* YFOV ; 
     static const char* PARALLEL ; 

     Camera(int width=1024, int height=768, float near=0.1f, float far=10000.f, float yfov=60.f, bool parallel=false) 
       :
         m_changed(true)
     {
         setSize(width, height);
         setPixelFactor(1); 

         setNearClip(1e-6, 1e6);
         setFarClip(1e-6, 1e6);
         setYfovClip(1.f, 179.f);

         setNear(near);
         setFar(far);
         setYfov(yfov);
         setParallel(parallel);
     } 

     glm::mat4 getProjection();
     glm::mat4 getPerspective();
     glm::mat4 getOrtho();
     glm::mat4 getOrthoScaled();
     glm::mat4 getOrthoScaled2();
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




     void near_to( float x, float y, float dx, float dy )
     {
         setNear(m_near + m_near*dy );
         //printf("Camera::near_to %10.3f \n", m_near);
     }

     void far_to( float x, float y, float dx, float dy )
     {
         setFar(m_far + m_far*dy );
         //printf("Camera::far_to %10.3f \n", m_far);
     }

     void yfov_to( float x, float y, float dx, float dy )
     {
         setYfov(m_yfov + 50.*dy) ;
         //printf("Camera::yfov_to %10.3f \n", m_yfov);
     }

     void setNear(float near)
     {
         if(      near < m_nearclip[0] )  m_near = m_nearclip[0] ;
         else if( near > m_nearclip[1] )  m_near = m_nearclip[1] ;
         else                             m_near = near ;

         m_changed = true ; 
     }
     float getNear()
     {
         return m_near ;
     }
     void setFar(float far)
     {
         if(      far < m_farclip[0] )  m_far = m_farclip[0] ;
         else if( far > m_farclip[1] )  m_far = m_farclip[1] ;
         else                           m_far = far ;
         m_changed = true ; 
     }
     float getFar()
     {
         return m_far ;
     }
     void setYfov(float yfov)
     {
         if(      yfov < m_yfovclip[0] )  m_yfov = m_yfovclip[0] ;
         else if( yfov > m_yfovclip[1] )  m_yfov = m_yfovclip[1] ;
         else                             m_yfov = yfov ;
         m_changed = true ; 
     }
     float getYfov()
     {
         return m_yfov ;
     }

     void setSize(int width, int height )
     {
         m_size[0] = width ;
         m_size[1] = height ;
         m_changed = true ; 
     }

     void setPixelFactor(unsigned int factor)
     {
         m_pixel_factor = factor ; 
         m_changed = true ; 
     }

     void setNearClip(float _min, float _max)
     {
         m_nearclip[0] = _min ;  
         m_nearclip[1] = _max ;  
     }
     void setFarClip(float _min, float _max)
     {
         m_farclip[0] = _min ;  
         m_farclip[1] = _max ;  
     }
     void setYfovClip(float _min, float _max)
     {
         m_yfovclip[0] = _min ;  
         m_yfovclip[1] = _max ;  
     }

     float getAspect() // width/height (> 1 for landscape)
     {
         return (float)m_size[0]/(float)m_size[1] ;  
     }


     void setParallel(bool parallel)
     {
         m_parallel = parallel ;
         m_changed = true ; 
     }
     bool getParallel()
     {
          return m_parallel ; 
     }
     float getTanYfov()
     {
         return tan( m_yfov*0.5f*M_PI/180.f );
     } 
     float getTop()
     {
         return m_near * getTanYfov();
     }
     float getBottom()
     {
         return -m_near * getTanYfov();
     }
     float getLeft()
     {
         return getAspect() * getBottom() ;
     } 
     float getRight()
     {
         return getAspect() * getTop() ;
     } 

     void Print(const char* msg="Camera::Print");
     void Summary(const char* msg="Camera::Summary");


     unsigned int getWidth()
     {
         return m_size[0]; 
     }
     unsigned int getHeight()
     {
         return m_size[1]; 
     }

     unsigned int getPixelWidth()
     {
         return m_size[0]*m_pixel_factor; 
     }
     unsigned int getPixelHeight()
     {
         return m_size[1]*m_pixel_factor; 
     }
     unsigned int getPixelFactor()
     {
         return m_pixel_factor ; 
     }



  private:

    
     int   m_size[2] ;
     int   m_pixel_factor ; 
     float m_nearclip[2] ;
     float m_farclip[2] ;
     float m_yfovclip[2] ;

     float m_near ;
     float m_far ;
     float m_yfov ;

     bool m_parallel ; 
     bool m_changed ; 

};


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



