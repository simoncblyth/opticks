#pragma once

#include <vector>
#include <glm/glm.hpp>  

// ggeo-
#include "GVector.hh"


template<typename T>
class NPY ; 


class MultiViewNPY ; 

class Camera ;
class View ;
class Light ;
class Trackball ; 
class Clipper ; 
class Cfg ;
class Scene ; 
class Animator ; 

#include "Configurable.hh"

class Composition : public Configurable {
  public:
      static const char* PRINT ;  
      static const char* SELECT ;  
      static const char* RECSELECT ;  

      friend class Interactor ;   
      friend class Bookmarks ;   
  public:
 
      Composition();

      virtual ~Composition();
      void nextMode(unsigned int modifiers);
      unsigned int tick();
      unsigned int getCount();

  private:
      void init();
      void initAnimator();

  public:
      // Configurable : for bookmarking 
      static bool accepts(const char* name);
      void configure(const char* name, const char* value);
      std::vector<std::string> getTags();
      void set(const char* name, std::string& s);
      std::string get(const char* name);

      // for cli/live updating 
      void configureI(const char* name, std::vector<int> values );
      void configureS(const char* name, std::vector<std::string> values);
      void gui();

  public: 
      void setCenterExtent(gfloat4 ce, bool autocam=false); // effectively points at what you want to look at 
      void setCenterExtent(glm::vec4& ce, bool autocam=false); // effectively points at what you want to look at 
      void setDomainCenterExtent(gfloat4 ce);               // typically whole geometry domain
      void setTimeDomain(gfloat4 td);
      void setColorDomain(gfloat4 cd);
      //void setLightPositionEye(gfloat4 lp);

  public:
      // avaiable as uniform inside shaders allowing GPU-side selections 
      void setSelection(glm::ivec4 sel);
      void setSelection(std::string sel);
      glm::ivec4& getSelection();

  public:
      void setRecSelect(glm::ivec4 sel);
      void setRecSelect(std::string sel);
      glm::ivec4& getRecSelect();

  public:
      void setParam(glm::vec4 par);
      void setParam(std::string par);
      glm::vec4&  getParam();

  public:
      void setFlags(glm::ivec4 flags);
      void setFlags(std::string flags);
      glm::ivec4& getFlags();

  public:
      void setPick(glm::ivec4 pick);
      void setPick(std::string pick);
      glm::ivec4& getPick();

  public:
      void setTarget(unsigned int target);
      void setScene(Scene* scene);
      void addConfig(Cfg* cfg);

  public: 
      void home();
      void update();

  public: 
      void test_getEyeUVW();
      void getEyeUVW(glm::vec3& eye, glm::vec3& U, glm::vec3& V, glm::vec3& W);
      void getEyeUVW_no_trackball(glm::vec3& eye, glm::vec3& U, glm::vec3& V, glm::vec3& W);
      void getLookAt(glm::mat4& lookat);

  //private: 
  public: 
      // private getters of residents : usable by friend class
      Camera*    getCamera(); 
      Trackball* getTrackball(); 
      View*      getView(); 
      Light*     getLight(); 
      Clipper*   getClipper(); 
      Scene*     getScene(); 

      
      void setCamera(Camera* camera);
      void setView(View* view);

  public: 
      // getters of inputs 
      glm::mat4& getModelToWorld();
      float getExtent();
      float getNear();
      float getFar();


  public: 
      unsigned int getWidth();
      unsigned int getHeight();
      unsigned int getPixelWidth(); // width*pixel_factor
      unsigned int getPixelHeight();
      unsigned int getPixelFactor();
      void setSize(unsigned int width, unsigned int height, unsigned int pixelfactor=1);
      void setSize(glm::uvec4 size); // x, y will be scaled down by the pixelfactor

  public:
      glm::vec3 unProject(unsigned int x, unsigned int y, float z);

  public:
      // getters of the derived properties : need to call update first to make them current
      glm::vec4& getGaze();
      glm::vec4& getCenterExtent();
      glm::vec4& getDomainCenterExtent();
      glm::vec4& getTimeDomain();
      glm::vec4& getColorDomain();
      glm::vec4& getLightPosition();
      glm::vec4& getLightDirection();
      float&     getGazeLength();
      glm::mat4& getWorld2Eye();  // ModelView  including trackballing
      float*     getWorld2EyePtr();  // ModelView  including trackballing
      glm::mat4& getEye2World();
      glm::mat4& getWorld2Camera();
      glm::mat4& getCamera2World();
      glm::mat4& getEye2Look();
      glm::mat4& getLook2Eye();

   public:
      // ModelViewProjection including trackballing
      glm::mat4& getWorld2Clip();   
      float*     getWorld2ClipPtr();  
      glm::mat4& getDomainISNorm();
      glm::mat4& getWorld2ClipISNorm();      // with initial domain_isnorm
      float*     getWorld2ClipISNormPtr();   

   public:
      float*     getIdentityPtr(); 
      glm::mat4& getProjection(); 

  private:
      glm::mat4& getTrackballing(); 
      glm::mat4& getITrackballing(); 

  public:
      int        getClipMode();
      glm::vec4& getClipPlane();
      float*     getClipPlanePtr();
  public:
      // analog to NumpyEvt for the axis
      void setAxisData(NPY<float>* axis_data);
      NPY<float>* getAxisData();
      MultiViewNPY* getAxisAttr();
      void dumpAxisData(const char* msg="Composition::dumpAxisData");

  private:
      // inputs 
      glm::mat4 m_model_to_world ; 
      float     m_extent ; 
      glm::vec4 m_center_extent ; 
      glm::vec4 m_domain_center_extent ; 
      glm::mat4 m_domain_isnorm ;     
      glm::vec4 m_domain_time ; 
      glm::vec4 m_domain_color ; 
      glm::vec4 m_light_position  ; 
      glm::vec4 m_light_direction ; 

  private:
      glm::ivec4 m_recselect ;
      glm::ivec4 m_selection ;
      glm::ivec4 m_flags ;
      glm::ivec4 m_pick ;
      glm::vec4  m_pick_f ; // for inputing pick using float slider 
      glm::vec4  m_param ;
      bool       m_animated ; 

  private:
      // residents
      Animator*   m_animator ; 
      Camera*    m_camera ;
      Trackball* m_trackball ;
      View*      m_view ;
      Light*     m_light ;
      Clipper*   m_clipper ;
      unsigned int m_count ; 
      NPY<float>*  m_axis_data ; 
      MultiViewNPY* m_axis_attr ;

      // visitors
      Scene*       m_scene ; 

  private:
      // updated by *update* based on inputs and residents
      glm::vec4 m_viewport ; 
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
      glm::mat4 m_world2clip_isnorm ;     
      glm::mat4 m_projection ;     

      glm::mat4 m_trackballing ;     
      glm::mat4 m_itrackballing ;     

      glm::mat4 m_trackballrot ;     
      glm::mat4 m_itrackballrot ;     
      glm::mat4 m_trackballtra ;     
      glm::mat4 m_itrackballtra ;     

      glm::mat4 m_identity ;     


  public: 
      void Summary(const char* msg);
      void Details(const char* msg);



};      

inline Composition::Composition()
  :
  m_model_to_world(),
  m_extent(1.0f),
  m_center_extent(),
  m_recselect(), 
  m_selection(-INT_MAX,-INT_MAX,-INT_MAX,-INT_MAX),  // not 0, as that is liable to being meaningful
  m_pick( 1,0,0,0),      // initialize modulo scaledown to 1, 0 causes all invisible 
  m_param(25.f,0.f,0.f,0.f),   // x: arbitrary scaling of genstep length 
  m_animator(NULL),
  m_camera(NULL),
  m_trackball(NULL),
  m_view(NULL),
  m_light(NULL),
  m_clipper(NULL),
  m_count(0),
  m_axis_data(NULL),
  m_axis_attr(NULL),
  m_scene(NULL)
{
    init();
}



inline Camera* Composition::getCamera()
{
    return m_camera ;
}
inline View* Composition::getView()
{
    return m_view ;
}
inline Light* Composition::getLight()
{
    return m_light ;
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
    m_camera = camera ; 
}
inline void Composition::setView(View* view)
{
    m_view = view ; 
}
inline Scene* Composition::getScene()
{
    return m_scene ; 
}
inline void Composition::setScene(Scene* scene)
{
    m_scene = scene ; 
}
inline glm::vec4& Composition::getCenterExtent()
{
    return m_center_extent ; 
}
inline glm::vec4& Composition::getDomainCenterExtent()
{
    return m_domain_center_extent ; 
}
inline glm::vec4& Composition::getTimeDomain()
{
    return m_domain_time ; 
}
inline glm::vec4& Composition::getColorDomain()
{
    return m_domain_color ; 
}
inline glm::vec4& Composition::getLightPosition()
{
    return m_light_position ; 
}
inline glm::vec4& Composition::getLightDirection()
{
    return m_light_direction ; 
}





inline glm::mat4& Composition::getDomainISNorm()
{
    return m_domain_isnorm ; 
}


inline glm::ivec4& Composition::getRecSelect()
{
    return m_recselect ; 
}
inline glm::ivec4& Composition::getSelection()
{
    return m_selection ; 
}

inline glm::ivec4& Composition::getFlags()
{
    return m_flags ; 
}
inline glm::ivec4& Composition::getPick()
{
    return m_pick; 
}
inline glm::vec4& Composition::getParam()
{
    return m_param ; 
}
inline glm::mat4& Composition::getModelToWorld()
{
    return m_model_to_world ; 
}
inline float Composition::getExtent()
{
    return m_extent ; 
}



inline unsigned int Composition::getCount()
{
    return m_count ; 
}

inline NPY<float>* Composition::getAxisData()
{
    return m_axis_data ; 
}

inline MultiViewNPY* Composition::getAxisAttr()
{
    return m_axis_attr ; 
}


