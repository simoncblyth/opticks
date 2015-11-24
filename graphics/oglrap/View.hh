#pragma once

#include <glm/glm.hpp>  
#include <vector>
#include <string>
#include <cstdio>
#include <math.h>

#include "Configurable.hh"


class Animator ; 

class View : public Configurable {
public:
   static const char* EYE ; 
   static const char* LOOK ; 
   static const char* UP ; 

   View();
   void initAnimator();
   void nextMode(unsigned int modifiers);
   void tick();
   virtual ~View();

   void configureS(const char* name, std::vector<std::string> values);
   void gui();

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

   glm::vec4 getEye(const glm::mat4& m2w);
   glm::vec4 getLook(const glm::mat4& m2w);
   glm::vec4 getUp(const glm::mat4& m2w);
   glm::vec4 getGaze(const glm::mat4& m2w, bool debug=false);

   glm::mat4 getLookAt(const glm::mat4& m2w, bool debug=false);

   void Summary(const char* msg="View::Summary");
   void Print(const char* msg="View::Print");

   void getFocalBasis(const glm::mat4& m2w,  glm::vec3& e, glm::vec3& u, glm::vec3& v, glm::vec3& w);
   void getTransforms(const glm::mat4& m2w, glm::mat4& world2camera, glm::mat4& camera2world, glm::vec4& gaze );

public:
   float getDistanceToAxis();
   float getEyePhase();
   void updateEyePhase();
   void setEyePhase( float t);
public:
   bool hasChanged();
   void setChanged(bool changed); 

private:
   glm::vec3 m_eye ; 
   float     m_eye_phase ; 
   glm::vec3 m_look ; 
   glm::vec3 m_up ; 
   bool      m_changed ; 
   std::vector<glm::vec4> m_axes ; 
   Animator* m_animator ; 

};


inline View::View() : m_animator(NULL)
{
    home();
    initAnimator();

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
    updateEyePhase();

    m_look.x =  0.f ; 
    m_look.y =  0.f ; 
    m_look.z =  0.f ;

    m_up.x =  0.f ; 
    m_up.y =  0.f ; 
    m_up.z =  1.f ;

}






inline View::~View()
{
}





// TODO: generalize to handle a choice of rotation axis

inline float View::getEyePhase()
{
   // somewhat dodgy derived qyt 
    return m_eye_phase ; 
}



inline float View::getDistanceToAxis()
{
    return sqrt( m_eye.x*m_eye.x + m_eye.y*m_eye.y );
}

inline void View::setEyePhase(float t)
{
    float d = getDistanceToAxis() ;
    m_eye_phase = t  ;
    double s, c ;  
#ifdef __APPLE__
    __sincos( m_eye_phase*M_PI , &s, &c);
#else
    sincos(m_eye_phase*M_PI , &s, &c);
#endif
    m_eye.x = d*c ; 
    m_eye.y = d*s ;
    m_changed = true ; 
}




inline void View::setEye( float _x, float _y, float _z)
{
    m_eye.x = _x ;  
    m_eye.y = _y ;  
    m_eye.z = _z ;  
    updateEyePhase();
    m_changed = true ; 

    printf("View::setEye %10.3f %10.3f %10.3f \n", _x, _y, _z);
}  


inline void View::setLook(float _x, float _y, float _z)
{
    m_look.x = _x ;  
    m_look.y = _y ;  
    m_look.z = _z ;  
    m_changed = true ; 

    printf("View::setLook %10.3f %10.3f %10.3f \n", _x, _y, _z);
}

inline void View::setUp(  float _x, float _y, float _z)
{
    m_up.x = _x ;  
    m_up.y = _y ;  
    m_up.z = _z ;  
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






