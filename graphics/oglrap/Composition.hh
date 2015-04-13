#ifndef COMPOSITION_H 
#define COMPOSITION_H

#include <vector>
#include <glm/glm.hpp>  

class Camera ;
class View ;
class Trackball ; 

class Composition {
  public:
 
      Composition();
      virtual ~Composition();

  public: 
      void setModelToWorld_Extent(float* m2w, float extent);
      void setSize(unsigned int width, unsigned int height);

  public:
      unsigned int getWidth();
      unsigned int getHeight();

  public: 
      void test_getEyeUVW();
      void getEyeUVW(glm::vec3& eye, glm::vec3& U, glm::vec3& V, glm::vec3& W);
      void getEyeUVW_no_trackball(glm::vec3& eye, glm::vec3& U, glm::vec3& V, glm::vec3& W);
      void getLookAt(glm::mat4& lookat);

      void update();

  public: 
      // getters of residents 
      Camera* getCamera(); 
      Trackball* getTrackball(); 
      View* getView(); 
  public: 
      // getters of inputs 
      glm::mat4& getModelToWorld();
      float getExtent();

  public:
      // getters of the derived properties
      glm::vec4& getGaze();
      float&     getGazeLength();
      glm::mat4& getWorld2Eye();  // ModelView  including trackballing
      glm::mat4& getEye2World();
      glm::mat4& getWorld2Camera();
      glm::mat4& getCamera2World();
      glm::mat4& getEye2Look();
      glm::mat4& getLook2Eye();
      glm::mat4& getWorld2Clip();  // ModelViewProjection  including trackballing
      glm::mat4& getProjection(); 
      glm::mat4& getTrackballing(); 
      glm::mat4& getITrackballing(); 

  private:
      // inputs 
      glm::mat4 m_model_to_world ; 
      float     m_extent ; 

  private:
      // residents
      Camera*    m_camera ;
      Trackball* m_trackball ;
      View*      m_view ;

  private:
      // updated by *update* based on inputs and residents
      glm::vec4 m_gaze ; 
      float     m_gazelength ;
      glm::mat4 m_world2eye ;     
      glm::mat4 m_eye2world ;     
      glm::mat4 m_world2camera ; 
      glm::mat4 m_camera2world ; 
      glm::mat4 m_eye2look ;     
      glm::mat4 m_look2eye ;     
      glm::mat4 m_world2clip ;     
      glm::mat4 m_projection ;     
      glm::mat4 m_trackballing ;     
      glm::mat4 m_itrackballing ;     

  public: 
      // housekeeping
      static const char* PRINT ;  
      void configureI(const char* name, std::vector<int> values );
      void Summary(const char* msg);
      void Details(const char* msg);



};      

#endif
