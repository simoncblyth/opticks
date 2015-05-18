#pragma once

#include <glm/glm.hpp>  
#include <vector>
#include <string>

#include "Configurable.hh"

class View : public Configurable {
public:
   static const char* EYE ; 
   static const char* LOOK ; 
   static const char* UP ; 

   View();
   virtual ~View()
   {
   }

   static bool accepts(const char* name);
   void configure(const char* name, const char* value);
   void configureS(const char* name, std::vector<std::string> values);


 public:
   std::vector<std::string> getTags();
   void set(const char* name, std::string& xyz);
   std::string get(const char* name);

   void setEye( float _x, float _y, float _z)
   {
      m_eye_x = _x ;  
      m_eye_y = _y ;  
      m_eye_z = _z ;  
   }  
   void setLook(float _x, float _y, float _z)
   {
      m_look_x = _x ;  
      m_look_y = _y ;  
      m_look_z = _z ;  
   }
   void setUp(  float _x, float _y, float _z)
   {
      m_up_x = _x ;  
      m_up_y = _y ;  
      m_up_z = _z ;  
   } 

   glm::vec4 getEye();
   glm::vec4 getLook();
   glm::vec4 getUp();
   glm::vec4 getGaze();
 
   glm::vec4 getEye(const glm::mat4& m2w);
   glm::vec4 getLook(const glm::mat4& m2w);
   glm::vec4 getUp(const glm::mat4& m2w);
   glm::vec4 getGaze(const glm::mat4& m2w, bool debug=false);

   glm::mat4 getLookAt(const glm::mat4& m2w, bool debug=false);

   void Summary(const char* msg="View::Summary");
   void Print(const char* msg="View::Print");

   void getFocalBasis(const glm::mat4& m2w,  glm::vec3& e, glm::vec3& u, glm::vec3& v, glm::vec3& w);
   void getTransforms(const glm::mat4& m2w, glm::mat4& world2camera, glm::mat4& camera2world, glm::vec4& gaze );


private:
   float m_eye_x ;
   float m_eye_y ;
   float m_eye_z ;

   float m_look_x ;
   float m_look_y ;
   float m_look_z ;

   float m_up_x ;
   float m_up_y ;
   float m_up_z ;

};



inline View::View()  : 
      m_eye_x(-1.0f),
      m_eye_y(-1.0f),
      m_eye_z(0.0f),
      m_look_x(0.0f),
      m_look_y(0.0f),
      m_look_z(0.0f),
      m_up_x(0.0f),
      m_up_y(0.0f),
      m_up_z(1.0f)
{
}


