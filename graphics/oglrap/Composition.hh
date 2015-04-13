#ifndef COMPOSITION_H 
#define COMPOSITION_H

#include <vector>
#include <glm/glm.hpp>  

class Camera ;
class View ;
class Trackball ; 

class Composition {
  public:
      static const char* PRINT ;  
 
      Composition();
      virtual ~Composition();
      void configureI(const char* name, std::vector<int> values );

  public: 
      void setModelToWorld(float* m2w);
      void setSize(unsigned int width, unsigned int height);
      void defineViewMatrices(glm::mat4& ModelView, glm::mat4& ModelViewInverse, glm::mat4& ModelViewProjection);
      void Summary(const char* msg);
      void Details(const char* msg);

      unsigned int getWidth();
      unsigned int getHeight();

  public: 
      Camera* getCamera(); 
      Trackball* getTrackball(); 
      View* getView(); 
      glm::mat4& getModelToWorld();

      void test_getEyeUVW();
      void getEyeUVW(glm::vec3& eye, glm::vec3& U, glm::vec3& V, glm::vec3& W);
      void getEyeUVW_no_trackball(glm::vec3& eye, glm::vec3& U, glm::vec3& V, glm::vec3& W);
      void getLookAt(glm::mat4& lookat);
      void view_transform(const glm::vec3& eye, const glm::vec3& look, const glm::vec3& up, glm::mat4& world2camera, glm::mat4& camera2world );

      void update();

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
      Camera* m_camera ;
      Trackball* m_trackball ;
      View*     m_view ;
      glm::mat4 m_model_to_world ; 

  private:
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




};      

#endif
