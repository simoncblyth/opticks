#pragma once


#include <vector>
#include <string>

#include <glm/fwd.hpp>  

//#define VIEW_DEBUG
#include "NConfigurable.hpp"
#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

template <typename T> class NPY ; 

/**
View
======


**/

class OKCORE_API View : public NConfigurable {
public:
   typedef enum { STANDARD, FLIGHTPATH, INTERPOLATED, ORBITAL, TRACK, NUM_VIEW_TYPE } View_t ; 

   static const char* STANDARD_ ; 
   static const char* FLIGHTPATH_ ; 
   static const char* INTERPOLATED_ ; 
   static const char* ORBITAL_ ; 
   static const char* TRACK_ ; 
   static const char* TypeName( View_t v ); 


   static const char* PREFIX ; 
   virtual const char* getPrefix();

   static const char* EYE ; 
   static const char* LOOK ; 
   static const char* UP ; 

   static View* FromArrayItem( NPY<float>* flightpath, unsigned i ); 

   View(View_t type=STANDARD);
   virtual ~View();

   bool isStandard();
   bool isInterpolated();
   bool isOrbital();
   bool isTrack();

   void configureF(const char* name, std::vector<float> values);
   void configureI(const char* name, std::vector<int> values);
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
   void setEyeX( float _x);
   void setEyeY( float _y);
   void setEyeZ( float _z);


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
   virtual void reset();
   virtual void tick();
   virtual void nextMode(unsigned int modifiers);
   virtual bool isActive(); // always false, used in InterpolatedView
   virtual bool hasChanged();

public:
   glm::mat4 getLookAt(const glm::mat4& m2w, bool debug=false);

   void Summary(const char* msg="View::Summary");
   void Print(const char* msg="View::Print");

   void getFocalBasis(const glm::mat4& m2w,  glm::vec3& e, glm::vec3& u, glm::vec3& v, glm::vec3& w);
   void getTransforms(const glm::mat4& m2w, glm::mat4& world2camera, glm::mat4& camera2world, glm::vec4& gaze );
public:
   void setChanged(bool changed); 

private:
   View_t    m_type ; 
   glm::vec3 m_eye ; 
   glm::vec3 m_look ; 
   glm::vec3 m_up ; 
   bool      m_changed ; 
   std::vector<glm::vec4> m_axes ; 

};

#include "OKCORE_TAIL.hh"


