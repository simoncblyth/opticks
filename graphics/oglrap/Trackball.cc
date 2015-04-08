#include "Trackball.hh"
#include "Common.hh"
#include "stdio.h"
#include <math.h>  
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>



glm::vec3 Trackball::getTranslation()
{
    return glm::vec3(m_x, m_y, m_z);
}

glm::mat4 Trackball::getTranslationMatrix()
{
    return glm::translate(  glm::mat4(1.0f), glm::vec3(m_x, m_y, m_z));
}

glm::mat4 Trackball::getCombinedMatrix()
{
    return glm::translate( getOrientationMatrix(), glm::vec3(m_x, m_y, m_z));
}



void Trackball::setOrientation(float _theta, float _phi)
{
    float theta = _theta*M_PI/180. ;
    float phi   = _phi*M_PI/180. ;

    glm::quat xrot(cos(0.5*theta),sin(0.5*theta),0,0);
    glm::quat zrot(cos(0.5*phi),  0,0,sin(0.5*phi));
    glm::quat q = xrot + zrot ; 
    setOrientation(q);
}


glm::quat Trackball::getOrientation()
{
   return m_orientation ;  
}

glm::mat4 Trackball::getOrientationMatrix()
{
   return glm::toMat4(m_orientation) ;  
}

void Trackball::setOrientation(glm::quat& q)
{
    m_orientation = q ; 
}

void Trackball::drag_to(float x, float y, float dx, float dy)
{
    printf("Trackball::drag_to %10.3f %10.3f %10.3f %10.3f \n", x, y, dx, dy);

    glm::quat dragrot = rotate(x,y,dx,dy);

    glm::quat qrot = getOrientation();
    qrot += dragrot ;    // perturb orientation by dragrot  

    setOrientation(qrot);
}


void Trackball::Summary(const char* msg)
{
    print(getOrientation(), msg);
    print(getOrientationMatrix(), msg);
}


glm::quat Trackball::rotate(float x, float y, float dx, float dy)
{
    //
    // p0, p1 are positions of screen before and after coordinates 
    // projected onto a deformed virtual trackball

    glm::vec3 p0(x   ,    y, project(m_radius,x,y)); 
    glm::vec3 p1(x+dx, y+dy, project(m_radius,x+dx,y+dy)); 
    
    // axis of rotation        
    glm::vec3 axis = glm::cross(p0, p1);

    // angle of rotation
    float t = glm::clamp(glm::length(p1-p0)/(2.*m_radius), -1., 1. ) ;
    float phi = 2.0 * asin(t) ;

    return glm::angleAxis( phi, axis );
}
 



float Trackball::project(float r, float x, float y)
{
  /*
        Project an x,y pair onto a sphere of radius r OR a hyperbolic sheet
        if we are away from the center of the sphere.

        For points inside xy circle::

                  d^2 = x^2 + y^2

                    d < r / sqrt(2)   

                  d^2 < r^2 / 2 

            x^2 + y^2 < r^2 / 2 
   

        determine z from::

                z^2 + d^2 = r^2 

        So are projecting onto sphere the center of which is on the screen plane.
  */      
 
  float z, t, d ;
  d = sqrt(x*x + y*y);
  if (d < r * 0.70710678118654752440)
  {     
      z = sqrt(r*r - d*d);
  }
  else
  {   
     t = r / 1.41421356237309504880 ;
     z = t*t / d  ;
  }
  return z;
}


