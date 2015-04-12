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
      void defineViewMatrices(glm::mat4& ModelView, glm::mat4& ModelViewProjection);
      void Summary(const char* msg);

      unsigned int getWidth();
      unsigned int getHeight();

  public: 
      Camera* getCamera(); 
      Trackball* getTrackball(); 
      View* getView(); 
      glm::mat4& getModelToWorld();

      // hmm doesnt incorporate the trackball ?
      // presumably can use a more matrix style approach to do that
      void getEyeUVW(glm::vec3& eye, glm::vec3& U, glm::vec3& V, glm::vec3& W);

  private:
      Camera* m_camera ;
      Trackball* m_trackball ;
      View*     m_view ;
      glm::mat4 m_model_to_world ; 

};      

#endif
