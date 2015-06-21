#pragma once

#include <glm/glm.hpp>  
#include <vector>
#include <string>
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

private:
   glm::vec3 m_eye ; 
   float     m_eye_phase ; 
   glm::vec3 m_look ; 
   glm::vec3 m_up ; 

   Animator* m_animator ; 

};


inline View::View() : m_animator(NULL)
{
    home();
    initAnimator();
}


inline void View::home()
{
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

inline void View::setEye( float _x, float _y, float _z)
{
    m_eye.x = _x ;  
    m_eye.y = _y ;  
    m_eye.z = _z ;  
    updateEyePhase();
}  

inline void View::updateEyePhase()
{
   // atan2 : Principal arc tangent of y/x, in the interval [-pi,+pi] radians 
   // so eye phase range is -1:1
    m_eye_phase = atan2(m_eye.y,m_eye.x)/(1.0f*M_PI);
}
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
    float s, c ;  
    __sincosf( m_eye_phase*M_PI , &s, &c);
    m_eye.x = d*c ; 
    m_eye.y = d*s ;
}


inline void View::setLook(float _x, float _y, float _z)
{
    m_look.x = _x ;  
    m_look.y = _y ;  
    m_look.z = _z ;  
}

inline void View::setUp(  float _x, float _y, float _z)
{
    m_up.x = _x ;  
    m_up.y = _y ;  
    m_up.z = _z ;  
} 




