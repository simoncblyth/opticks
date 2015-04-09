#include "Trackball.hh"
#include "Common.hh"
#include "stdio.h"
#include <math.h>  
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>


const char* Trackball::RADIUS = "radius" ;
const char* Trackball::ORIENTATION = "orientation" ;
const char* Trackball::TRANSLATE = "translate" ;
const char* Trackball::TRANSLATEFACTOR = "translatefactor" ;


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


void Trackball::configureF(const char* name, std::vector<float> values)
{
     if(values.empty())
     {
         printf("Trackball::configureF %s no values \n", name);
     }
     else         
     {
         float vlast = values.back() ;

#ifdef VERBOSE
         printf("Trackball::configureF %s : %lu values : ", name, values.size());
         for(size_t i=0 ; i < values.size() ; i++ ) printf("%10.3f ", values[i]);
         printf(" : vlast %10.3f \n", vlast );
#endif

         if(      strcmp(name, RADIUS) ==  0)          setRadius(vlast);
         else if( strcmp(name, TRANSLATEFACTOR) ==  0) setTranslateFactor(vlast);
         else
              printf("Trackball::configureF ignoring unknown parameter %s : %10.3f \n", name, vlast); 
     }
}
 

void Trackball::configureS(const char* name, std::vector<std::string> values)
{
     if(values.empty())
     {
         printf("Trackball::configureS %s no values \n", name);
     }
     else         
     {
         std::string  vlast = values.back() ;

#ifdef VERBOSE
         printf("Trackball::configureS %s : %lu values : ", name, values.size());
         for(size_t i=0 ; i < values.size() ; i++ ) printf("%20s ", values[i]);
         printf(" : vlast %20s \n", vlast );
#endif

         if(      strcmp(name, TRANSLATE)   ==  0) setTranslate(vlast);
         else if( strcmp(name, ORIENTATION) ==  0) setOrientation(vlast);
         else
              printf("Trackball::configureS ignoring unknown parameter %s : %s \n", name, vlast.c_str()); 
     }
}
 
void Trackball::setOrientation(std::string _tp)
{
    std::vector<std::string> tp;
    boost::split(tp, _tp, boost::is_any_of(","));

    if(tp.size() == 2 )
    {
        float theta = boost::lexical_cast<float>(tp[0]); 
        float phi   = boost::lexical_cast<float>(tp[1]); 
        setOrientation(theta,phi);
    }
    else
    {
        printf("Trackball::setOrientation malformed _tp : %s \n", _tp.c_str() );
    }
}

void Trackball::setTranslate(std::string _xyz)
{
    std::vector<std::string> xyz;
    boost::split(xyz, _xyz, boost::is_any_of(","));

    if(xyz.size() == 3 )
    {
        float x = boost::lexical_cast<float>(xyz[0]); 
        float y = boost::lexical_cast<float>(xyz[1]); 
        float z = boost::lexical_cast<float>(xyz[2]); 
        setTranslate(x,y,z);
    }
    else
    {
        printf("Trackball::setTranslate malformed _xyz : %s \n", _xyz.c_str() );
    }
}




void Trackball::setOrientation(float _theta, float _phi)
{
    float theta = _theta*M_PI/180. ;
    float phi   = _phi*M_PI/180. ;

    glm::quat xrot(cos(0.5*theta),sin(0.5*theta),0,0);
    glm::quat zrot(cos(0.5*phi),  0,0,sin(0.5*phi));
    glm::quat q = xrot * zrot ; 
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
    m_orientation = q  ; 
}

void Trackball::drag_to(float x, float y, float dx, float dy)
{
    //printf("Trackball::drag_to %10.3f %10.3f %10.3f %10.3f \n", x, y, dx, dy);

    m_drag_count += 1 ; 

    glm::quat drag = rotate(x,y,dx,dy);

    glm::quat qrot = m_orientation * drag ;   // perturb orientation by drag rotation  

    if(m_drag_count > m_drag_renorm)
    {
        //print(qrot, "Trackball::drag_to before renorm");
        qrot = glm::normalize(qrot);         
        //print(qrot, "Trackball::drag_to after renorm");
        m_drag_count = 0 ;
    }

    setOrientation(qrot);
}


void Trackball::Summary(const char* msg)
{
    printf(" trackballradius  %10.3f \n", m_radius );
    print(getOrientation(), msg);
    //print(getOrientationMatrix(), msg);
}


glm::quat Trackball::rotate(float x, float y, float dx, float dy)
{
    // p0, p1 are positions of screen before and after coordinates 
    // projected onto a deformed virtual trackball

    glm::vec3 p0(x   ,    y, project(m_radius,x,y)); 
    glm::vec3 p1(x+dx, y+dy, project(m_radius,x+dx,y+dy)); 
   
    // axis of rotation        
    glm::vec3 axis = glm::cross(p1, p0);

    // angle of rotation
    float t = glm::clamp(glm::length(p1-p0)/(2.*m_radius), -1., 1. ) ;
    float phi = 2.0 * asin(t) ;

    glm::quat q = glm::angleAxis( phi, glm::normalize(axis) );

#ifdef DEBUG
    print(p0,  "Trackball::rotate p0");
    print(p1,  "Trackball::rotate p1");
    print(axis,"Trackball::rotate axis");
    printf("Trackball::rotate t %15.5f phi %15.5f \n", t, phi );
    print(q,   "Trackball::rotate q");
#endif

    return q ; 
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


