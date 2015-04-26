#ifndef COMPOSITION_H 
#define COMPOSITION_H

#include <vector>
#include <glm/glm.hpp>  

// ggeo-
#include "GVector.hh"


class Camera ;
class View ;
class Trackball ; 
class Clipper ; 
class Cfg ;

class Composition {

      friend class Interactor ;   
      friend class Bookmarks ;   
  public:
 
      Composition();
      virtual ~Composition();

  public: 
      //void setModelToWorld(float* m2w, bool debug=false); // effectively points at what you want to look at 
      void setCenterExtent(gfloat4 ce); // effectively points at what you want to look at 

      void setSize(unsigned int width, unsigned int height);
      void addConfig(Cfg* cfg);

  public: 
      void update();

  public: 
      void test_getEyeUVW();
      void getEyeUVW(glm::vec3& eye, glm::vec3& U, glm::vec3& V, glm::vec3& W);
      void getEyeUVW_no_trackball(glm::vec3& eye, glm::vec3& U, glm::vec3& V, glm::vec3& W);
      void getLookAt(glm::mat4& lookat);

  private: 
      // private getters of residents : usable by friend class
      Camera* getCamera(); 
      Trackball* getTrackball(); 
      View* getView(); 
      Clipper* getClipper(); 

      
      void setCamera(Camera* camera);
      void setView(View* view);

  public: 
      // getters of inputs 
      glm::mat4& getModelToWorld();
      float getExtent();
      float getNear();
      float getFar();
      unsigned int getWidth();
      unsigned int getHeight();
      unsigned int getPixelWidth(); // width*pixel_factor
      unsigned int getPixelHeight();
      unsigned int getPixelFactor();
      void setPixelFactor(unsigned int factor); // 2 for retina display

  public:
      // getters of the derived properties : need to call update first to make them current
      glm::vec4& getGaze();
      glm::vec4& getCenterExtent();
      float&     getGazeLength();
      glm::mat4& getWorld2Eye();  // ModelView  including trackballing
      float*     getWorld2EyePtr();  // ModelView  including trackballing
      glm::mat4& getEye2World();
      glm::mat4& getWorld2Camera();
      glm::mat4& getCamera2World();
      glm::mat4& getEye2Look();
      glm::mat4& getLook2Eye();
      glm::mat4& getWorld2Clip();     // ModelViewProjection  including trackballing
      float*     getWorld2ClipPtr();  // ModelViewProjection  including trackballing
      float*     getIdentityPtr(); 
      glm::mat4& getProjection(); 
      glm::mat4& getTrackballing(); 
      glm::mat4& getITrackballing(); 

  public:
      int        getClipMode();
      glm::vec4& getClipPlane();
      float*     getClipPlanePtr();

  private:
      // inputs 
      glm::mat4 m_model_to_world ; 
      float     m_extent ; 
      glm::vec4 m_center_extent ; 

  private:
      // residents
      Camera*    m_camera ;
      Trackball* m_trackball ;
      View*      m_view ;
      Clipper*   m_clipper ;

  private:
      // updated by *update* based on inputs and residents
      glm::vec4 m_gaze ; 
      glm::vec4 m_clipplane ; 
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
      glm::mat4 m_identity ;     

  public: 
      // housekeeping
      static const char* PRINT ;  
      void configureI(const char* name, std::vector<int> values );
      void Summary(const char* msg);
      void Details(const char* msg);



};      



inline Camera* Composition::getCamera()
{
    return m_camera ;
}
inline View* Composition::getView()
{
    return m_view ;
}
inline Trackball* Composition::getTrackball()
{
    return m_trackball ;
}
inline Clipper* Composition::getClipper()
{
    return m_clipper ;
}

inline void Composition::setCamera(Camera* camera)
{
    //delete m_camera ;
    m_camera = camera ; 
}
inline void Composition::setView(View* view)
{
    //delete m_view ;
    m_view = view ; 
}









#endif
