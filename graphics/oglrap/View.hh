#pragma once

#include <glm/glm.hpp>  
#include <vector>
#include <string>
#include <cstdio>
#include <math.h>

#include "NConfigurable.hpp"

//#define VIEW_DEBUG

class View : public NConfigurable {
public:
   static const char* PREFIX ; 
   virtual const char* getPrefix();

   static const char* EYE ; 
   static const char* LOOK ; 
   static const char* UP ; 

   View();

   void configureS(const char* name, std::vector<std::string> values);

 public:
   // Configurable
   
   static bool accepts(const char* name);
   void configure(const char* name, const char* value);
   std::vector<std::string> getTags();
   void set(const char* name, std::string& xyz);
   std::string get(const char* name);

 public:
   void home(); 

   void setEye( float _x, float _y, float _z);
   void setLook(float _x, float _y, float _z);
   void setUp(  float _x, float _y, float _z);
   void handleDegenerates();

   void setEye( glm::vec4& eye );
   void setLook( glm::vec4& look );
   void setUp(  glm::vec4& up);

   glm::vec4 getEye();
   glm::vec4 getLook();
   glm::vec4 getUp();
   glm::vec4 getGaze();
 
   float* getEyePtr();
   float* getLookPtr();
   float* getUpPtr();

public:
   // methods overridden in InterpolatedView
   virtual glm::vec4 getEye(const glm::mat4& m2w);
   virtual glm::vec4 getLook(const glm::mat4& m2w);
   virtual glm::vec4 getUp(const glm::mat4& m2w);
   virtual glm::vec4 getGaze(const glm::mat4& m2w, bool debug=false);
   virtual void tick();
   virtual void nextMode(unsigned int modifiers);
   virtual bool isActive(); // always false, used in InterpolatedView
   virtual bool hasChanged();
   virtual void gui();
public:
   glm::mat4 getLookAt(const glm::mat4& m2w, bool debug=false);

   void Summary(const char* msg="View::Summary");
   void Print(const char* msg="View::Print");

   void getFocalBasis(const glm::mat4& m2w,  glm::vec3& e, glm::vec3& u, glm::vec3& v, glm::vec3& w);
   void getTransforms(const glm::mat4& m2w, glm::mat4& world2camera, glm::mat4& camera2world, glm::vec4& gaze );
public:
   void setChanged(bool changed); 

private:
   glm::vec3 m_eye ; 
   glm::vec3 m_look ; 
   glm::vec3 m_up ; 
   bool      m_changed ; 
   std::vector<glm::vec4> m_axes ; 

};


inline View::View() 
{
    home();

    m_axes.push_back(glm::vec4(0,1,0,0));
    m_axes.push_back(glm::vec4(0,0,1,0));
    m_axes.push_back(glm::vec4(1,0,0,0));
}

inline bool View::hasChanged()
{
    return m_changed ; 
}

inline void View::setChanged(bool changed)
{
    m_changed = changed ; 
}

inline void View::home()
{
    m_changed = true ; 
    m_eye.x = -1.f ; 
    m_eye.y = -1.f ; 
    m_eye.z =  0.f ;

    m_look.x =  0.f ; 
    m_look.y =  0.f ; 
    m_look.z =  0.f ;

    m_up.x =  0.f ; 
    m_up.y =  0.f ; 
    m_up.z =  1.f ;
}

inline void View::setEye( float _x, float _y, float _z)
{
    m_eye.x = _x ;  
    m_eye.y = _y ;  
    m_eye.z = _z ;  

    handleDegenerates();
    m_changed = true ; 

#ifdef VIEW_DEBUG
    printf("View::setEye %10.3f %10.3f %10.3f \n", _x, _y, _z);
#endif
}  

inline void View::setLook(float _x, float _y, float _z)
{
    m_look.x = _x ;  
    m_look.y = _y ;  
    m_look.z = _z ;  

    handleDegenerates();
    m_changed = true ; 

#ifdef VIEW_DEBUG
    printf("View::setLook %10.3f %10.3f %10.3f \n", _x, _y, _z);
#endif
}

inline void View::setUp(  float _x, float _y, float _z)
{
    m_up.x = _x ;  
    m_up.y = _y ;  
    m_up.z = _z ;  

    handleDegenerates();
    m_changed = true ; 
} 

inline void View::setLook(glm::vec4& look)
{
    setLook(look.x, look.y, look.z );
}
inline void View::setEye(glm::vec4& eye)
{
    setEye(eye.x, eye.y, eye.z );
}
inline void View::setUp(glm::vec4& up)
{
    setUp(up.x, up.y, up.z );
}

