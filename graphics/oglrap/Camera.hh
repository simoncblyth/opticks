#ifndef CAMERA_H
#define CAMERA_H

#include <glm/glm.hpp>  
#include <math.h> 

class Camera {
  public:

     Camera(int width, int height, float near=0.1f, float far=10000.f, float yfov=60.f, bool parallel=false) 
     {
         setSize(width, height);

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
     glm::mat4 getFrustum();

     void setNear(float near)
     {
         if(      near < m_nearclip[0] )  m_near = m_nearclip[0] ;
         else if( near > m_nearclip[1] )  m_near = m_nearclip[1] ;
         else                             m_near = near ;
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
     }
     float getYfov()
     {
         return m_yfov ;
     }

     void setSize(int width, int height)
     {
         m_size[0] = width ;
         m_size[1] = height ;
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

     float getAspect()
     {
         return (float)m_size[0]/(float)m_size[1] ;  
     }


     void setParallel(bool parallel)
     {
          m_parallel = parallel ;
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

     void Summary(const char* msg="Camera::Summary");


  private:

    
     int   m_size[2] ;
     float m_nearclip[2] ;
     float m_farclip[2] ;
     float m_yfovclip[2] ;

     float m_near ;
     float m_far ;
     float m_yfov ;

     bool m_parallel ; 

};


#endif
