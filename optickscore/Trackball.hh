#pragma once

#include <vector>
#include <string>
#include <glm/fwd.hpp>  

#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"
#include "NConfigurable.hpp"

class OKCORE_API Trackball : public NConfigurable {
   public:
       static const char* PREFIX ;
       const char* getPrefix();
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
       float* getTranslationPtr();
       float getTranslationMin();
       float getTranslationMax();
   public:
       float* getRadiusPtr();
       float getRadiusMin();
       float getRadiusMax();
  public:
       float* getTFactorPtr();
       float getTFactorMin();
       float getTFactorMax();
  public:
       float getRadius();
       float getTranslateFactor(); 
       glm::quat getOrientation();
       std::string getOrientationString();
       glm::vec3 getTranslation();

       glm::mat4 getOrientationMatrix();
       glm::mat4 getTranslationMatrix();
       glm::mat4 getCombinedMatrix();

       float* getOrientationPtr();

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

#include "OKCORE_TAIL.hh"



