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

       void gui(); 
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
       void setRadiusClip(float _min, float _max);
       void setTranslateFactor(float tf);
       void setTranslateFactor(std::string tf);
       void setTranslateFactorClip(float _min, float _max);
       void setTranslate(float x, float y, float z);
       void setTranslate(std::string xyz);
       void setTranslateMax(float _max);
       void setOrientation(float theta, float phi);
       void setOrientation(glm::quat& q);
       void setOrientation(std::string tp);
       void setOrientation();

  public:
     bool hasChanged();
     void setChanged(bool changed); 

  public:
       float getRadius();
       float getTranslateFactor(); 
       glm::quat getOrientation();
       glm::vec3 getTranslation();

       glm::mat4 getOrientationMatrix();
       glm::mat4 getTranslationMatrix();
       glm::mat4 getCombinedMatrix();

       float* getOrientationPtr();
       float* getTranslationPtr();

       void getCombinedMatrices(glm::mat4& rt, glm::mat4& irt);
       void getOrientationMatrices(glm::mat4& rot, glm::mat4& irot);
       void getTranslationMatrices(glm::mat4& tra, glm::mat4& itra);

   public:
       void Summary(const char* msg="Trackball::Summary");

   private:
       glm::quat rotate(float x, float y, float dx, float dy);
       static float project(float r, float x, float y);

   private:
       glm::vec3 m_translate ; 
       float m_translate_max ; 

       float m_radius ;
       float m_translatefactor ;

       float m_radius_clip[2] ; 
       float m_translatefactor_clip[2] ;
     
       glm::quat m_orientation ; 

       int  m_drag_renorm ; 
       int  m_drag_count ; 

       // input qtys used to construct quaternion
       float m_theta_deg ; 
       float m_phi_deg   ; 

       bool m_changed ; 


};




inline Trackball::Trackball() :
          m_radius(1.f),
          m_translatefactor(1000.f),
          m_drag_renorm(97),
          m_drag_count(0),
          m_changed(true)
{
    home();

    setTranslateMax(1e5);
    setRadiusClip(0.1, 10.f);
    setTranslateFactorClip(10.f, 1e6);
}


inline void Trackball::setTranslateMax(float _max)
{
    m_translate_max = _max ; 
}

inline void Trackball::setRadiusClip(float _min, float _max)
{
    m_radius_clip[0] = _min ;  
    m_radius_clip[1] = _max ;  
}

inline void Trackball::setTranslateFactorClip(float _min, float _max)
{
    m_translatefactor_clip[0] = _min ;  
    m_translatefactor_clip[1] = _max ;  
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
    m_changed = true ; 
}
inline void Trackball::setTranslateFactor(float tf)
{
    m_translatefactor = tf ; 
    m_changed = true ; 
}
inline void Trackball::setTranslate(float x, float y, float z)
{
    m_translate.x = x;
    m_translate.y = y;
    m_translate.z = z;
    m_changed = true ; 
}
 
inline void Trackball::home()
{
    m_translate.x = 0.f ; 
    m_translate.y = 0.f ; 
    m_translate.z = 0.f ; 
   setOrientation(0.f, 0.f);
    m_changed = true ; 
}
inline void Trackball::zoom_to(float x, float y, float dx, float dy)
{
    m_translate.z += dy*m_translatefactor ;
    m_changed = true ; 
} 
inline void Trackball::pan_to(float x, float y, float dx, float dy)
{
    m_translate.x += dx*m_translatefactor ;
    m_translate.y += dy*m_translatefactor ;
    m_changed = true ; 
} 

inline bool Trackball::hasChanged()
{
    return m_changed ; 
}
inline void Trackball::setChanged(bool changed)
{
    m_changed = changed ; 
}


