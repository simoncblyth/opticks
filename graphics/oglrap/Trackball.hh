#pragma once
#include <glm/glm.hpp>  
#include <glm/gtc/quaternion.hpp>
#include <vector>
#include <string>

#include "Configurable.hh"

class Trackball : public Configurable {
   public:
       static const char* RADIUS ;
       static const char* ORIENTATION ;
       static const char* TRANSLATE ;
       static const char* TRANSLATEFACTOR ;

       Trackball();

   public:
       // Configurable
       static bool accepts(const char* name);
       void configure(const char* name, const char* value_);
       std::vector<std::string> getTags();
       void set(const char* name, std::string& s);
       std::string get(const char* name);

  public:
       void configureF(const char* name, std::vector<float> values);
       void configureS(const char* name, std::vector<std::string> values);

   public:
       void home();
       void zoom_to(float x, float y, float dx, float dy);
       void pan_to(float x, float y, float dx, float dy);
       void drag_to(float x, float y, float dx, float dy);

  public:
       void setRadius(float r);
       void setRadius(std::string r);
       void setTranslateFactor(float tf);
       void setTranslateFactor(std::string tf);
       void setTranslate(float x, float y, float z);
       void setTranslate(std::string xyz);
       void setOrientation(float theta, float phi);
       void setOrientation(glm::quat& q);
       void setOrientation(std::string tp);

  public:
       float getRadius();
       float getTranslateFactor(); 
       glm::quat getOrientation();
       glm::vec3 getTranslation();
       glm::mat4 getOrientationMatrix();
       glm::mat4 getTranslationMatrix();
       glm::mat4 getCombinedMatrix();

       void getCombinedMatrices(glm::mat4& rt, glm::mat4& irt);
       void getOrientationMatrices(glm::mat4& rot, glm::mat4& irot);
       void getTranslationMatrices(glm::mat4& tra, glm::mat4& itra);

   public:
       void Summary(const char* msg);

   private:
       glm::quat rotate(float x, float y, float dx, float dy);
       static float project(float r, float x, float y);

   private:
       float m_x ;
       float m_y ;
       float m_z ;

       float m_radius ;
       float m_translatefactor ;

       glm::quat m_orientation ; 

       int  m_drag_renorm ; 
       int  m_drag_count ; 


};




inline Trackball::Trackball() :
          m_radius(1.f),
          m_translatefactor(1000.f),
          m_drag_renorm(97),
          m_drag_count(0)
{
    home();
}


inline float Trackball::getRadius()
{
    return m_radius ; 
}
inline float Trackball::getTranslateFactor()
{
    return m_translatefactor  ; 
}


inline void Trackball::setRadius(float r)
{
    m_radius = r ; 
}
inline void Trackball::setTranslateFactor(float tf)
{
    m_translatefactor = tf ; 
}
inline void Trackball::setTranslate(float x, float y, float z)
{
    m_x = x;
    m_y = y;
    m_z = z;
}
 
inline void Trackball::home()
{
    m_x = 0.f ; 
    m_y = 0.f ; 
    m_z = 0.f ; 
    setOrientation(0.f, 0.f);
}
inline void Trackball::zoom_to(float x, float y, float dx, float dy)
{
    m_z += dy*m_translatefactor ;
} 
inline void Trackball::pan_to(float x, float y, float dx, float dy)
{
    m_x += dx*m_translatefactor ;
    m_y += dy*m_translatefactor ;
} 


