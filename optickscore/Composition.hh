#pragma once

#include <string>
#include <vector>

#include <glm/fwd.hpp>  


//class SCtrl ; 
// bcfg-
class BCfg ;

// npy-
template<typename T> class NPY ; 
class MultiViewNPY ; 
class NState ; 

// opticks-
class Camera ;
class OrbitalView ; 
class TrackView ; 
class Light ;
class Trackball ; 
class Clipper ; 

class ContentStyle; 
class RenderStyle; 
class GlobalStyle; 

class Animator ; 
class Bookmarks ; 
class FlightPath ; 
class OpticksEvent ; 


#include "NConfigurable.hpp"
#include "View.hh"

#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

/**

Composition
=============

Canonical m_composition instance is ctor resident of the top level OpticksHub m_hub (okg-). 

Encapsulates

* world frame center/extent *ce* of object of interest in geometry 
* camera parameters near, far, zoom
* viewpoint eye, look, up in model frame (ie relative to current ce of an object of interest)

Pointers to the canonical Composition instance are held by:

* AxisApp
* Frame
* GUI
* InstanceCuller
* Interactor  : for changing parameters
* OpticksViz
* Rdr
* Renderer
* Scene


Primary purpose of the composition instance is to supply uniforms to the shaders, 
as controlled by the Renderer (oglrap-)::

    504 void Renderer::update_uniforms()
    505 {
    506     if(m_composition)
    507     {
    508         m_composition->update() ;
    509         glUniformMatrix4fv(m_mv_location, 1, GL_FALSE,  m_composition->getWorld2EyePtr());
    510         glUniformMatrix4fv(m_mvp_location, 1, GL_FALSE, m_composition->getWorld2ClipPtr());
    511 
    512 
    513         glUniform4fv(m_param_location, 1, m_composition->getParamPtr());
    514 
    515         glUniform4fv(m_scanparam_location, 1, m_composition->getScanParamPtr());
    516         glm::vec4 sp = m_composition->getScanParam();
    517 


**/

class OKCORE_API Composition : public NConfigurable {
   public:
      friend class GUI ; 
   public:
      static const char* PREFIX ;
      const char* getPrefix();
   public:
      // see CompositionCfg.hh
      static const char* PRINT ;  
      static const char* SELECT ;  
      static const char* RECSELECT ;  
      static const char* PICKPHOTON ;  
      static const char* EYEW ; 
      static const char* LOOKW ; 
      static const char* UPW ; 

      friend class Interactor ;   
      friend class Bookmarks ;   
  public:
      static const char* WHITE_ ; 
      static const char* MAT1_ ; 
      static const char* MAT2_ ; 
      static const char* FLAG1_ ; 
      static const char* FLAG2_ ; 
      static const char* POL1_ ; 
      static const char* POL2_ ; 
  public:
      static const char* DEF_GEOMETRY_ ; 
      static const char* NRMCOL_GEOMETRY_ ; 
      static const char* VTXCOL_GEOMETRY_ ; 
      static const char* FACECOL_GEOMETRY_ ; 
  public:
      static const glm::vec3 X ; 
      static const glm::vec3 Y ; 
      static const glm::vec3 Z ; 
  public:
      Composition();
      void addConstituentConfigurables(NState* state);
      virtual ~Composition();
   public:
      //void setCtrl(SCtrl* ctrl ); 
      void command(const char* cmd) ;
   public:
      void setAnimatorPeriod(int period);
      Animator* getAnimator();
   public:
      // A_KEY
      void nextAnimatorMode(unsigned int modifiers);
      void commandAnimatorMode(const char* cmd);
   public:
      // V_KEY
      void nextRotatorMode(unsigned int modifiers);
   public:
      // T_KEY ?
      void nextViewMode(unsigned int modifiers);
      void commandViewMode(const char* cmd) ;
      void resetComposition();  // SHIFT+OPTION+T
      void changeView(unsigned int modifiers);
      unsigned int tick();
      unsigned int getCount();
   private:
      void swapView();
      void initAnimator();
      void initRotator();
   public:
      void setOrbitalViewPeriod(int ovperiod);
      void setTrackViewPeriod(int tvperiod);
      void setTrack(NPY<float>* track);
      OrbitalView* makeOrbitalView();
      TrackView* makeTrackView();
   public:
       void scrub_to(float x, float y, float dx, float dy); // Interactor:K scrub_mode
   public:
      // U_KEY 
      void nextViewType(unsigned int modifiers);
      void setViewType(View::View_t type);
      View::View_t getViewType();
   private:
       void applyViewType();
   public:
       // M_KEY
       typedef enum { WHITE, MAT1, MAT2, FLAG1, FLAG2, POL1, POL2, NUM_COLOR_STYLE } ColorStyle_t ;
       static const char* getColorStyleName(Composition::ColorStyle_t style);
       const char* getColorStyleName();
       void nextColorStyle();
       void setColorStyle(Composition::ColorStyle_t style);
       Composition::ColorStyle_t getColorStyle();
   public:
       // D_KEY
       void nextCameraStyle(unsigned modifiers);
       bool hasNoRasterizedRender() const ;    // EQUIRECTANGULAR camera currently only works in raytrace
   public:
       typedef enum { DEF_NORMAL, FLIP_NORMAL, NUM_NORMAL_STYLE } NormalStyle_t ;
       void nextNormalStyle();   // changes m_nrmparam.x
       void setNormalStyle(Composition::NormalStyle_t style);
       Composition::NormalStyle_t getNormalStyle();
   public:
       // E_KEY
       typedef enum { DEF_GEOMETRY, NRMCOL_GEOMETRY, VTXCOL_GEOMETRY, FACECOL_GEOMETRY, NUM_GEOMETRY_STYLE } GeometryStyle_t ;
       static const char* getGeometryStyleName(Composition::GeometryStyle_t style);
       const char* getGeometryStyleName();
       void nextGeometryStyle();   // changes m_nrmparam.y
       void commandGeometryStyle(const char* cmd) ;
       void setGeometryStyle(Composition::GeometryStyle_t style);
       Composition::GeometryStyle_t getGeometryStyle();
   public:
       typedef enum { SHOW, HIDE, NUM_PICKPHOTON_STYLE } PickPhotonStyle_t ;
       void nextPickPhotonStyle();
       void setPickPhotonStyle(Composition::PickPhotonStyle_t style);
       Composition::PickPhotonStyle_t getPickPhotonStyle();
   public:
        // RenderStyle : O_KEY 
        void nextRenderStyle(unsigned modifiers); 
        void commandRenderStyle(const char* cmd); 
        bool isProjectiveRender() const ;
        bool isRaytracedRender() const ;
        bool isCompositeRender() const ;
        void setRaytraceEnabled(bool enable) ; 
   public:
        // Clipper : C_KEY 
        void nextClipperStyle(); 
        void commandClipper(const char* cmd);  // C0:off C1:on 
   public:
        typedef enum { NONE, ENABLED, NUM_PIXELTIME_STYLE } PixelTimeStyle_t ; 
        void nextPixelTimeStyle(unsigned modifiers);
        unsigned getPixelTimeStyle() const ; 
        float*   getPixelTimeScalePtr() ; 
        float    getPixelTimeScale() const ; 
        float    getPixelTimeScaleMin() const ; 
        float    getPixelTimeScaleMax() const ; 
   public:
        void commandCameraNear(const char* cmd);  // N0:default N1:def/2 N2:def/4 
   public:
        // ContentStyle : B_KEY
        ContentStyle* getContentStyle() const ;
        void          nextContentStyle(); 
        void          commandContentStyle(const char* cmd);
   public:
        // GlobalStyle : Q_KEY 
        GlobalStyle* getGlobalStyle() const ;
        bool* getGlobalModePtr();
        bool* getGlobalVecModePtr();
        void nextGlobalStyle();
        void commandGlobalStyle(const char* cmd);
   public:
        // "." PERIOD_KEY    
        void nextPauseStyle();
  private:
      void init();
      void initAxis();
  public:
      // NConfigurable : for bookmarking 
      static bool accepts(const char* name);
      void configure(const char* name, const char* value);
      std::vector<std::string> getTags();
      void set(const char* name, std::string& s);
      std::string get(const char* name);

      // for cli/live updating : BCfg binding  
      void configureF(const char* name, std::vector<float> values );
      void configureI(const char* name, std::vector<int> values );
      void configureS(const char* name, std::vector<std::string> values);

  public: 
      void aim(glm::vec4& ce, bool verbose=false);
      void setCenterExtent(const glm::vec4& ce, bool aim=false); // effectively points at what you want to look at 
      //void setFaceTarget(unsigned int face_index, unsigned int solid_index, unsigned int mesh_index);
      //void setFaceRangeTarget(unsigned int face_index0, unsigned int face_index1, unsigned int solid_index, unsigned int mesh_index);
  public: 
      void setDomainCenterExtent(const glm::vec4& ce); // typically whole geometry domain
      void setColorDomain(const glm::uvec4& cd);
      void setTimeDomain(const glm::vec4& td);
  public:
      // avaiable as uniform inside shaders allowing GPU-side selections 
      void setSelection(const glm::ivec4& sel);
      void setSelection(std::string sel);
      glm::ivec4& getSelection();

  public:
      void setRecSelect(const glm::ivec4& sel);
      void setRecSelect(std::string sel);
      glm::ivec4& getRecSelect();

  public:
      void setPickPhoton(const glm::ivec4& pp);
      void setPickPhoton(std::string pp);
      glm::ivec4& getPickPhoton();
      int* getPickPtr();
  public:
      void setPickFace(const glm::ivec4& pf);
      void setPickFace(std::string pf);
      glm::ivec4& getPickFace();

  public:
      void setColorParam(const glm::ivec4& cp);
      void setColorParam(std::string cp);
      glm::ivec4& getColorParam();
      int* getColorParamPtr();
  public:
      void setParam(const glm::vec4& par);
      void setParam(std::string par);
      glm::vec4&  getParam();
      float*      getParamPtr();

  public:
      glm::vec4&  getScanParam();
      float*      getScanParamPtr();

  public:
      //void setNrmParam(glm::ivec4 par);
      //void setNrmParam(std::string par);
      glm::ivec4&  getNrmParam();
      int*         getNrmParamPtr();

  public:
      void setFlags(const glm::ivec4& flags);
      void setFlags(std::string flags);
      glm::ivec4& getFlags();

  public:
      void setPick(const glm::ivec4& pick);
      void setPick(std::string pick);
      glm::ivec4& getPick();
  public:
      void setEvt(OpticksEvent* evt);
      OpticksEvent* getEvt();
  public:
      void addConfig(BCfg* cfg);
  public:
      void setLookAngle(float phi);
      float* getLookAnglePtr();
  public:
      void setLookW(const glm::vec4& lookw);
      void setLookW(std::string lookw);
  public:
      void setEye(float _x, float _y, float _z);
  public:
      float getEyeX() const ;  
      float getEyeY() const ;  
      float getEyeZ() const ;  
      void setEyeX(float _x);
      void setEyeY(float _y);
      void setEyeZ(float _z);

      void setEyeW(const glm::vec4& eyew);
      void setEyeW(std::string eyew);
  public:
      void setUpW(const glm::vec4& upw);
      void setUpW(std::string upw);
  public:
      void setEyeGUI(const char* cmd);
      void setUpGUI(const char* cmd);
      void setEyeGUI(const glm::vec3& gui);
  public: 
      void home();
      void update();

  public: 
      void test_getEyeUVW();
      unsigned getCameraType() const ;   // formerly getParallel
      unsigned getTimeTrace() const ;
      void getEyeUVW(glm::vec3& eye, glm::vec3& U, glm::vec3& V, glm::vec3& W, glm::vec4& ZProj);
      void getEyeUVW_no_trackball(glm::vec3& eye, glm::vec3& U, glm::vec3& V, glm::vec3& W);
      void getLookAt(glm::mat4& lookat);
  public: 
      glm::vec3 getNDC(const glm::vec4& position_world) const ;
      glm::vec3 getNDC2(const glm::vec4& position_world) const ;
      float getNDCDepth(const glm::vec4& position_world);
      float getClipDepth(const glm::vec4& position_world);
  private: 
      // invoked from Interactor 
      void commitView();
  public: 
      // private getters of residents : usable by friend class

      Camera*    getCamera(); 
      Trackball* getTrackball(); 
      View*      getView(); 
      Light*     getLight(); 
      Clipper*   getClipper(); 
     
      void setCameraType(unsigned cameratype);
      void setCamera(Camera* camera);
      void setView(View* view);
      void resetView();
      void setBookmarks(Bookmarks* bookmarks);
      void setFlightPath(FlightPath* flightpath);
  public: 
      // getters of inputs 
      const glm::vec4& getLODCut() const ;
      glm::mat4& getModelToWorld();
      glm::mat4& getWorldToModel();
      float getExtent();
      float getNear();
      float getFar();
  public:
      // position of the observer "Viewpoint" and the observed "Lookpoint" using m_eye_to_world/m_world_to_eye
      glm::vec4 transformWorldToEye(const glm::vec4& world) const ;
      glm::vec4 transformEyeToWorld(const glm::vec4& eye) const ;
      glm::vec4 getLookpoint();
      glm::vec4 getViewpoint();
      glm::vec4 getUpdir();
  public: 
      unsigned int getWidth();
      unsigned int getHeight();
      unsigned int getPixelWidth(); // width*pixel_factor
      unsigned int getPixelHeight();
      unsigned int getPixelFactor();
      void setSize(unsigned int width, unsigned int height, unsigned int pixelfactor=1);
      void setSize(const glm::uvec4& size); // x, y will be scaled down by the pixelfactor
      void setFramePosition(const glm::uvec4& position);
      glm::uvec4& getFramePosition();
  public:
      glm::vec3 unProject(unsigned int x, unsigned int y, float z);

  public:
      // getters of the derived properties : need to call update first to make them current
      glm::vec4& getGaze();
      glm::vec4& getCenterExtent();
      glm::vec4& getDomainCenterExtent();
      glm::vec4& getTimeDomain();
      glm::uvec4& getColorDomain();
      glm::vec4& getLightPosition();
      float*     getLightPositionPtr();
      glm::vec4& getLightDirection();
      float*     getLightDirectionPtr();
      float&     getGazeLength();
      glm::mat4& getWorld2Eye();  // ModelView  including trackballing
      float*     getWorld2EyePtr();  // ModelView  including trackballing
      glm::mat4& getEye2World();
      glm::mat4& getWorld2Camera();
      glm::mat4& getCamera2World();
      glm::mat4& getEye2Look();
      glm::mat4& getLook2Eye();
   public:
      std::string getEyeString();
      std::string getLookString();
      std::string getGazeString();
   public:
      bool hasChanged(); // based on View, Camera, Trackball
      bool hasChangedGeometry(); // excludes event animation
      void setChanged(bool changed);
   public:
      // ModelViewProjection including trackballing
      glm::mat4& getWorld2Clip();   
      float*     getWorld2ClipPtr();  
      glm::mat4& getDomainISNorm();
      glm::mat4& getWorld2ClipISNorm();      // with initial domain_isnorm
      float*     getWorld2ClipISNormPtr();   
      glm::mat4& getProjection(); 
      float*     getProjectionPtr();  
   public:
      float*     getIdentityPtr(); 

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
  public:
      void dumpFrustum(const char* msg="Composition::dumpFrustum") const ;
      void dumpCorners(const char* msg="Composition::dumpCorners") const ;
      void dumpPoints(const std::vector<glm::vec4>& world, const std::vector<std::string>& labels) const  ;
      void getCorners(std::vector<glm::vec4>& corners, std::vector<std::string>& labels) const  ; 
  private:
      // inputs 
      glm::vec4 m_lodcut ; 
      glm::mat4 m_model2world ; 
      glm::mat4 m_world2model ; 
      float     m_extent ; 
      glm::vec4 m_center_extent ; 
      glm::vec4 m_domain_center_extent ; 
      glm::mat4 m_domain_isnorm ;     
      glm::vec4 m_domain_time ; 
      glm::uvec4 m_domain_color ; 
      glm::vec4 m_light_position  ; 
      glm::vec4 m_light_direction ; 

  private:
      glm::ivec4 m_pickphoton ;  // see CompositionCfg.hh 
      glm::ivec4 m_pickface ;
      glm::ivec4 m_recselect ;
      glm::ivec4 m_colorparam ;
      glm::ivec4 m_selection ;
      glm::ivec4 m_flags ;
      glm::ivec4 m_pick ;
      glm::vec4  m_pick_f ; // for inputing pick using float slider 
      glm::vec4  m_param ;
      glm::vec4  m_scanparam ;
      glm::ivec4  m_nrmparam ;
      bool       m_animated ; 

  private:
      // residents
      Animator*   m_animator ; 
      Animator*   m_rotator ; 
      Camera*    m_camera ;
      Trackball* m_trackball ;
      Bookmarks* m_bookmarks ; 
      FlightPath* m_flightpath ; 

      View*      m_view ;
      View*      m_standard_view ;
      View::View_t  m_viewtype ; 
      int         m_animator_period ; 
      int         m_ovperiod ; 
      int         m_tvperiod ; 
      NPY<float>* m_track ; 

      Light*     m_light ;
      Clipper*   m_clipper ;

      ContentStyle*  m_content_style ; 
      RenderStyle*   m_render_style ; 
      GlobalStyle*   m_global_style ; 
      bool           m_paused ;   

      unsigned int m_count ; 
      NPY<float>*  m_axis_data ; 
      MultiViewNPY* m_axis_attr ;
      bool          m_changed ; 

  private:
      // visitors
      OpticksEvent*  m_evt ; 
      //SCtrl*         m_ctrl ; 
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

      glm::mat4 m_lookrotation ;     
      glm::mat4 m_ilookrotation ;
      float     m_lookphi ;  // degrees
     
      glm::mat4 m_trackballing ;     
      glm::mat4 m_itrackballing ;     

      glm::mat4 m_trackballrot ;     
      glm::mat4 m_itrackballrot ;     
      glm::mat4 m_trackballtra ;     
      glm::mat4 m_itrackballtra ;     

      glm::mat4 m_identity ;     
      glm::vec4 m_axis_x ; 
      glm::vec4 m_axis_y ; 
      glm::vec4 m_axis_z ; 
      glm::vec4 m_axis_x_color ; 
      glm::vec4 m_axis_y_color ; 
      glm::vec4 m_axis_z_color ; 

      std::string   m_command ; 
      unsigned int  m_command_length ; 

  private:
      glm::uvec4 m_frame_position ; 
  private:
      unsigned  m_pixeltime ; 
      float     m_pixeltime_scale ; 
      float     m_pixeltime_scale_min ; 
      float     m_pixeltime_scale_max ; 
  public: 
      void Summary(const char* msg);
      void Details(const char* msg);

};      
#include "OKCORE_TAIL.hh"


