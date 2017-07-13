#pragma once


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




#include <vector>
#include <string>
#include <glm/fwd.hpp>  

#include "NConfigurable.hpp"
#include "OKCORE_API_EXPORT.hh"

class OKCORE_API Camera : public NConfigurable  {
   public:
       static const char* PREFIX ;
  public:
     static const char* PRINT ; 
     static const char* NEAR_ ; 
     static const char* FAR_ ; 
     static const char* ZOOM ; 
     static const char* SCALE ; 

     static const char* PARALLEL ; 

     Camera(int width=1024, int height=768, float basis=1000.f ) ;

     glm::mat4 getProjection();
     glm::mat4 getPerspective();
     glm::mat4 getOrtho();
     glm::mat4 getFrustum();

     void fillZProjection(glm::vec4& zProj);


     bool hasChanged();
     void setChanged(bool changed); 

   public:
       typedef enum { PERSPECTIVE_CAMERA, OTHOGRAPHIC_CAMERA, NUM_CAMERA_STYLE } Style_t ;
       void nextStyle(unsigned modifiers);
       void setStyle(Camera::Style_t style);
       Camera::Style_t getStyle();

 public:
     // NConfigurable realization
     const char* getPrefix();
     void configure(const char* name, const char* value);
     std::vector<std::string> getTags();
     std::string get(const char* name);
     void set(const char* name, std::string& xyz);

  public:
     static bool accepts(const char* name);

     // NB dont overload these as it confuses boost::bind
     void configureF(const char* name, std::vector<float> values);
     void configureI(const char* name, std::vector<int> values);
     void configureS(const char* name, std::vector<std::string> values);

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
     float getNearMin();
     float getNearMax();
     float getFarMin();
     float getFarMax();
     float getZoomMin();
     float getZoomMax();
     float getScaleMin();
     float getScaleMax();
  public:
     float* getNearPtr();
     float* getFarPtr();
     float* getZoomPtr();
     float* getScalePtr();
     bool* getParallelPtr();
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


