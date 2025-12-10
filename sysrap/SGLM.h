#pragma once
/**
SGLM : Header Only Viz Math giving ray tracing basis and rasterization projection transforms
===============================================================================================

Critical usage for ray trace rendering done in CSGOptiX::prepareRenderParam

SGLM.h is a single header that is replacing a boatload of classes
used by old OpticksCore okc : Composition, View, Camera, ...
Using this enabled CSGOptiX to drop dependency on okc, npy, brap
and instead depend only on QUDARap, SysRap, CSG.

Normal inputs WH, CE, EYE, LOOK, UP are held in static variables with envvar defaults
These can be changed with static methods before instanciating SGLM.
NB it will usually be too late for setenv in code to influence SGLM
as the static initialization would have happened already

* TODO: reincarnate animated interpolation between view bookmarks
* TODO: provide persistency into ~16 quad4 for debugging view/cam/projection state


SGLM::desc
-------------

NB : to dump view param from interactive viz use "P" key invoking SGLM::desc


DONE : rasterized and raytrace render consistency
-------------------------------------------------

Using get_transverse_scale proportional to get_near_abs
and longitudinally using get_near_abs from updateEyeBasis/updateProjection
for raytrace/rasterized succeeds to get the two perspective
renders to match closely. This follows the technique used in okc/Camera.cc.

* TODO: regain orthographic, above changes for perspective matching have messed that up
* TODO: flipping between raytrace and raster with C:CUDA key looses quaternion rotation causing jump back


SGLM.h tests
--------------

SGLMTest.cc
   check a few statics, standardly built

SGLM_test.{sh,cc}
   standalone test for a few SGLM methods

SGLM_set_frame_test.{sh,cc}
   loads sframe sets into SGLM and dumps

SGLM_frame_targetting_test.{sh,cc}
   compares SGLM A,B from two different center_extent sframe a,b


Review coordinate systems, following along the below description
-----------------------------------------------------------------

* https://unspecified.wordpress.com/2012/06/21/calculating-the-gluperspective-matrix-and-other-opengl-matrix-maths/
* https://learnopengl.com/Getting-started/Camera
* https://feepingcreature.github.io/math.html

OpenGL coordinate systems
~~~~~~~~~~~~~~~~~~~~~~~~~~

Right hand systems:

* +X : right
* +Y : up
* +Z : towards camera
* -Z : into scene


Object
   vertices relative to center of Model
World
   relative to one world origin
Eye
   relative to camera

   * vertices are transformed into Eye Coordinates by the model-view matrix

Clip
   funny coordinates : that get transformed by "divide-by-w" into NDC coordinates

   * shader pipeline (vertex or geometry shader) outputs Clip coordinates, that
     OpenGL does the “divide-by-w” step to givew NDC coordinates

   * the ".w" of clip coordinates often set to "-z" as trick to do perspective Divide


NDC/Viewport
   normalized device coordinates : on screen position coordinates

   * (x,y)
   * (-1,-1) : lower left
   * (+1,+1) : upper right
   * z=-1 : nearest point in depth buffer
   * z=+1 : farthest point in depth buffer

   The z values are mapped on to the depth buffer space by the projection matrix.
   Thats why zNear,ZFar settings are important.

Screen
   (x,y) coordinate on screen in pixels

   * (0,0) lower left pixel
   * (w,h) upper right pixel


**/

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <array>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/string_cast.hpp>

#include "scuda.h"
#include "sqat4.h"
#include "SCenterExtentFrame.h"
#include "SCAM.h"
#include "SBAS.h"
#include "SBitSet.h"

#include "SScene.h"
#include "stree.h"

#include "SGLM_Arcball.h"

#include "sfr.h"     // formerly sframe.h
#include "SCE.h"     // moving from sframe to SCE

#include "ssys.h"
#include "sstr.h"
#include "NP.hh"

#include "SYSRAP_API_EXPORT.hh"

#include "SRecord.h"
#include "SGen.h"
#include "SCMD.h"
#include "SGLM_Modifiers.h"
#include "SGLM_Parse.h"





// inputs to projection matrix
struct SYSRAP_API lrbtnf
{
    float left ;
    float right ;
    float bottom ;
    float top ;
    float near ;
    float far ;

    // see http://www.songho.ca/opengl/gl_projectionmatrix.html
    float A_frustum() const ;
    float B_frustum() const ;
    float A_ortho() const ;
    float B_ortho() const ;
    std::string desc() const ;
};



/**
SGLM_Toggle
-------------

**/

struct SYSRAP_API SGLM_Toggle
{
    bool zoom = false ;
    bool tmin = false ;
    bool tmax = false ;
    bool lrot = false ;
    bool cuda = false ;
    bool norm = false ;
    bool time = false ;
    bool stop = false ;
    std::string desc() const ;
};

inline std::string SGLM_Toggle::desc() const
{
    std::stringstream ss ;
    ss << "SGLM_Toggle::desc"
       << " zoom:" << ( zoom ? "Y" : "N" )
       << " tmin:" << ( tmin ? "Y" : "N" )
       << " tmax:" << ( tmax ? "Y" : "N" )
       << " lrot:" << ( lrot ? "Y" : "N" )
       << " cuda:" << ( cuda ? "Y" : "N" )
       << " norm:" << ( norm ? "Y" : "N" )
       << " time:" << ( time ? "Y" : "N" )
       << " stop:" << ( stop ? "Y" : "N" )
       ;
    std::string str = ss.str();
    return str ;
}


struct SYSRAP_API SGLM_Option
{
    bool A = false ;
    bool B = false ;
    bool G = false ;
    bool M = true ;
    bool O = true ;
    std::string desc() const ;
};

inline std::string SGLM_Option::desc() const
{
    std::stringstream ss ;
    ss << "SGLM_Option::desc"
       << " A:" << ( A ? "Y" : "N" )
       << " B:" << ( B ? "Y" : "N" )
       << " G:" << ( G ? "Y" : "N" )
       << " M:" << ( M ? "Y" : "N" )
       << " O:" << ( O ? "Y" : "N" )
       ;
    std::string str = ss.str();
    return str ;
}






struct SYSRAP_API SGLM : public SCMD
{
    static SGLM* INSTANCE ;
    static SGLM* Get();

    static constexpr const char* kWH = "WH" ;
    static constexpr const char* kCE = "CE" ;
    static constexpr const char* kEYE = "EYE" ;
    static constexpr const char* kLOOK = "LOOK" ;
    static constexpr const char* kUP = "UP" ;
    static constexpr const char* kZOOM = "ZOOM" ;
    static constexpr const char* kTMIN = "TMIN" ;
    static constexpr const char* kTMAX = "TMAX" ;
    static constexpr const char* kCAM = "CAM" ;
    static constexpr const char* kNEARFAR = "NEARFAR" ;
    static constexpr const char* kFOCAL = "FOCAL" ;
    static constexpr const char* kFULLSCREEN = "FULLSCREEN" ;
    static constexpr const char* kESCALE = "ESCALE" ;
    static constexpr const char* kEXTENT_FUDGE = "EXTENT_FUDGE" ;

    static constexpr const char* kVIZMASK = "VIZMASK" ;
    static constexpr const char* kTRACEYFLIP = "TRACEYFLIP" ;
    static constexpr const char* kLEVEL = "SGLM_LEVEL" ;
    static constexpr const char* kTIMESCALE = "TIMESCALE" ;

    static constexpr const char* kT0 = "T0" ;
    static constexpr const char* kT1 = "T1" ;
    static constexpr const char* kTT = "TT" ;
    static constexpr const char* kTN = "TN" ;  // int:number of render loop time bumps to go from T0 to T1




    static constexpr const char* _SGLM_DESC = "SGLM_DESC" ;
    static constexpr const char* __setTreeScene_DUMP = "SGLM__setTreeScene_DUMP" ;
    static constexpr const char* __init_time_DUMP = "SGLM__init_time_DUMP" ;


    // static defaults, some can be overridden in the instance
    static glm::ivec2 WH ;

    static glm::vec4 CE ; // CE IS GEOMETRY NOT VIEW RELATED BUT ITS EXPEDIENT TO BE HERE
    static glm::vec4 EYE ;
    static glm::vec4 LOOK ;
    static glm::vec4 UP ;


    static float ZOOM ;
    static float TMIN ;
    static float TMAX ;
    static int   CAM ;
    static int   NEARFAR ;
    static int   FOCAL ;
    static int   FULLSCREEN ;
    static int   ESCALE ;
    static float EXTENT_FUDGE ;
    static uint32_t VIZMASK ;
    static int   TRACEYFLIP ;
    static int   LEVEL ;

    static float TIMESCALE ;

    // record time range
    static float  T0 ;
    static float  T1 ;
    static float  TT ;
    static int    TN ;


    static void SetWH( int width, int height );
    static void SetCE( float x, float y, float z, float extent );
    static void SetEYE( float x, float y, float z );
    static void SetLOOK( float x, float y, float z );
    static void SetUP( float x, float y, float z );

    static void SetZOOM( float v );
    static void SetTMIN( float v );
    static void SetTMAX( float v );

    static void IncZOOM( float v );
    static void IncTMIN( float v );
    static void IncTMAX( float v );

    static void SetCAM( const char* cam );
    static void SetNEARFAR( const char* nearfar );
    static void SetFOCAL( const char* focal );
    static void SetFULLSCREEN( const char* fullscreen );
    static void SetESCALE( const char* escale );
    static void SetVIZMASK( const char* vizmask );
    static void SetTRACEYFLIP( const char* traceyflip );

    bool is_vizmask_set(unsigned ibit) const ;


    // querying of static defaults
    static std::string DescInput() ;
    static int Width() ;
    static int Height() ;
    static int Width_Height() ;

    static float Aspect();
    static const char* CAM_Label() ;
    static const char* NEARFAR_Label() ;
    static const char* FOCAL_Label() ;
    static const char* FULLSCREEN_Label() ;
    static const char* ESCALE_Label() ;
    static std::string VIZMASK_Label() ;
    static const char* TRACEYFLIP_Label() ;

    static void Copy(float* dst, const glm::vec3& src );
    static void Copy(float* dst, const glm::vec4& src );

    // member methods
    std::string descInput() const ;

    SGLM();
    void init();

    bool SGLM_DESC ;

    stree*  tree ;
    SScene* scene ;
    void setTreeScene( stree* _tree, SScene* _scene );

    void handle_frame_hop(int wanted_frame_idx);

    void setLookRotation(float angle_deg, glm::vec3 axis );
    void setLookRotation( const glm::vec2& a, const glm::vec2& b );

    void setEyeRotation(float angle_deg, glm::vec3 axis );
    void setEyeRotation( const glm::vec2& a, const glm::vec2& b );

    void setDepthTest(int _depth_test);
    int depthTest() const ;

    void cursor_moved_action( const glm::vec2& a, const glm::vec2& b, unsigned modifiers );
    void key_pressed_action( unsigned modifiers );

    void home();
    std::string descEyeShift() const ;
    static std::string DescQuat( const glm::quat& q );


    void tcam();
    void toggle_traceyflip();
    void toggle_rendertype();

    static void Command(const SGLM_Parse& parse, SGLM* gm, bool dump);
    int command(const char* cmd);


    sfr moi_fr = {} ;
    sfr fr = {} ;  // CAUTION: SEvt also holds an sframe used for input photon targetting

    static constexpr const char* _DUMP = "SGLM__set_frame_DUMP" ;
    void set_frame( const sfr& fr );
    int get_frame_idx() const ;
    bool has_frame_idx(int idx) const ;
    const std::string& get_frame_name() const ;

    float extent() const ;
    float tmin_abs() const ;
    float tmax_abs() const ;

    bool rtp_tangential ;
    void set_rtp_tangential( bool rtp_tangential_ );

    bool extent_scale ;
    void set_extent_scale( bool extent_scale_ );

    // matrices taken from fr or derived from ce when fr only holds identity
    glm::mat4 model2world ;
    glm::mat4 world2model ;
    int initModelMatrix_branch ;

    void initModelMatrix();  // depends on ce, unless non-identity m2w and w2m matrices provided in frame
    std::string descModelMatrix() const ;

    // world frame View converted from static model frame
    // initELU

    void  initELU();   // depends on CE and EYE, LOOK, UP
    void updateGaze();
    std::string descELU() const ;

    std::vector<glm::vec3> axes ;

    glm::vec3 eye ;
    glm::vec3 look ;
    glm::vec3 up ;

    // updateGaze
    glm::vec3 gaze ;
    glm::mat4 eye2look ;
    glm::mat4 look2eye ;

    glm::quat q_lookrot ;
    glm::quat q_eyerot ;
    glm::vec3 eyeshift  ;

    int       depth_test ;
    int       home_count ;

    float     get_escale_() const ;
    glm::mat4 get_escale() const ;
    float getGazeLength() const ;
    float getGazeCrossUp() const ;
    void avoidDegenerateBasisByChangingUp();

    void updateNearFar();
    std::string descNearFar() const ;

    // results from updateEyeSpace

    glm::vec3 forward_ax ;
    glm::vec3 right_ax ;
    glm::vec3 top_ax ;
    glm::mat4 rot_ax ;
    glm::mat4 world2camera ;
    glm::mat4 camera2world ;

    void updateEyeSpace();
    std::string descEyeSpace() const ;

    // results from updateEyeBasis
    glm::vec3 u ;
    glm::vec3 v ;
    glm::vec3 w ;
    glm::vec3 wnorm ;
    glm::vec3 e ;


    void updateEyeBasis();
    static std::string DescEyeBasis( const glm::vec3& E, const glm::vec3& U, const glm::vec3& V, const glm::vec3& W );
    std::string descEyeBasis() const ;




    // modes
    int  cam ;
    int  nearfar ;
    int  focal ;
    int  fullscreen ;
    uint32_t vizmask ;
    int   traceyflip ;
    int   rendertype ;


    float nearfar_manual ;
    float focal_manual ;
    float near ;
    float far ;
    float orthographic_scale ;


    std::vector<std::string> log ;



    void set_near( float near_ );
    void set_far( float far_ );
    void set_near_abs( float near_abs_ );
    void set_far_abs(  float far_abs_ );

    float get_near() const ;
    float get_far() const ;
    float get_near_abs() const ;
    float get_far_abs() const ;

    std::string descFrame() const ;
    std::string descBasis() const ;
    std::string descProj() const ;
    std::string descProjection() const ;
    std::string descComposite() const ;

    lrbtnf    proj ;
    glm::mat4 projection ;
    glm::vec4 zproj ;

    glm::mat4 MV ;
    glm::mat4 IMV ;

    float*    MV_ptr ;
    glm::mat4 MVP ;    // aka world2clip
    float*    MVP_ptr ;
    glm::mat4 IDENTITY ;
    float* IDENTITY_ptr ;
    std::string title ;

    void updateTitle();
    void left_right_bottom_top_near_far(lrbtnf& p) const ;

    static constexpr const char* _updateProjection_DEBUG = "SGLM__updateProjection_DEBUG" ;
    void updateProjection();

    static void FillZProjection(  glm::vec4& _zproj, const glm::mat4& _proj );
    float zdepth_pos( const glm::tvec4<float>& p_eye ) const ;
    float zdepth0(    const float& z_eye ) const ;
    float zdepth1(    const float& z_eye ) const ;
    float zproj_A() const ;
    float zproj_B() const ;

    static void FillAltProjection(glm::vec4& _aproj, const glm::mat4& _proj );

    float get_transverse_scale() const ;




    void updateComposite();


    template<typename T> void ce_corners_world( std::vector<glm::tvec4<T>>& v_world ) const ;
    template<typename T> void ce_midface_world( std::vector<glm::tvec4<T>>& v_world ) const ;

    template<typename T>
    static void Apply_XF( std::vector<glm::tvec4<float>>& v_out, const std::vector<glm::tvec4<T>>& v_in, const glm::tmat4x4<float>& XF, bool flip );

    template<typename T>
    void apply_MV(  std::vector<glm::tvec4<float>>& v_eye,  const std::vector<glm::tvec4<T>>& v_world, bool flip ) const ;
    template<typename T>
    void apply_MVP( std::vector<glm::tvec4<float>>& v_clip, const std::vector<glm::tvec4<T>>& v_world, bool flip ) const ;
    template<typename T>
    void apply_P(   std::vector<glm::tvec4<float>>& v_clip, const std::vector<glm::tvec4<T>>& v_eye  , bool flip ) const ;

    std::string desc_MVP() const ;
    std::string desc_MVP_ce_corners() const ;
    std::string desc_MV_P_MVP_ce_corners() const ;
    std::string desc_MVP_ce_midface() const ;
    static bool IsClipped(const glm::vec4& _ndc );

    void set_nearfar_mode(const char* mode);
    void set_focal_mode(const char* mode);

    const char* get_nearfar_mode() const ;
    const char* get_focal_mode() const ;

    void set_nearfar_manual(float nearfar_manual_);
    void set_focal_manual(float focal_manual_);

    float get_nearfar_basis() const ;
    float get_focal_basis() const ;

    void save(const char* dir, const char* stem) const ;
    void writeDesc(const char* dir, const char* name="SGLM__writeDesc", const char* ext=".log" ) const ;
    std::string desc() const ;


    void dump() const ;
    void update();
    void constrain() const ;


    void addlog( const char* label, float value       ) ;
    void addlog( const char* label, const char* value ) ;
    std::string descLog() const ;


    template <typename T> static T ato_( const char* a );
    template <typename T> static void Str2Vector( std::vector<T>& vec, const char* uval );
    template <typename T> static void GetEVector(std::vector<T>& vec, const char* key, const char* fallback );
    template <typename T> static std::string Present(std::vector<T>& vec);
    template <typename T> static std::string Present(const T* tt, int num);

    static std::string Present(const glm::ivec2& v, int wid=6 );
    static std::string Present(const float v, int wid=10, int prec=3);
    static std::string Present(const glm::vec2& v, int wid=10, int prec=3);
    static std::string Present(const glm::vec3& v, int wid=10, int prec=3);
    static std::string Present(const glm::vec4& v, int wid=10, int prec=3);
    static std::string Present(const float4& v,    int wid=10, int prec=3);
    static std::string Present(const glm::mat4& m, int wid=10, int prec=3);

    template<typename T> static std::string Present_(const glm::tmat4x4<T>& m, int wid=10, int prec=3);

    static void GetEVec(glm::vec3& v, const char* key, const char* fallback );
    static void GetEVec(glm::vec4& v, const char* key, const char* fallback );

    template <typename T> static T SValue(const char* uval );
    template <typename T> static T EValue(const char* key, const char* fallback );
    static glm::ivec2 EVec2i(const char* key, const char* fallback);
    static glm::vec3 EVec3(const char* key, const char* fallback);
    static glm::vec4 EVec4(const char* key, const char* fallback, float missing );
    static glm::vec4 SVec4(const char* str, float missing );
    static glm::vec3 SVec3(const char* str, float missing );

    template<typename T> static glm::tmat4x4<T> DemoMatrix(T scale);


    SRecord* ar ;
    SRecord* br ;
    SGen*    gs ;

    void setRecord( SRecord* ar, SRecord* br );
    void setGenstep( SGen* gs );

    bool enabled_time_bump = true ;
    bool enabled_time_halt = false ;
    glm::vec4 timeparam = {} ;
    const float* timeparam_ptr ;


    void init_time();
    void reset_time();
    void reset_time_TT();
    void toggle_time_halt();

    std::string desc_time() const ;

    float get_t0() const ;
    float get_t1() const ;
    float get_ts() const ;
    int get_tn() const ;


    float get_time() const ;
    bool in_timerange(float t) const ;
    void set_time( float t );
    void time_bump();
    void inc_time( float dy );

    SGLM_Toggle toggle = {} ;
    SGLM_Option option = {} ;

    void renderloop_head();
    void renderloop_tail();

};

SGLM* SGLM::INSTANCE = nullptr ;
SGLM* SGLM::Get(){  return INSTANCE ? INSTANCE : new SGLM  ; }

glm::ivec2 SGLM::WH = EVec2i(kWH,"1920,1080") ;

glm::vec4  SGLM::CE = EVec4(kCE,"0,0,0,100", 100.f) ;

glm::vec4  SGLM::EYE  = EVec4(kEYE, "-1,-1,0,1", 1.f) ;
glm::vec4  SGLM::LOOK = EVec4(kLOOK, "0,0,0,1" , 1.f) ;
glm::vec4  SGLM::UP  =  EVec4(kUP,   "0,0,1,0" , 0.f) ;

float      SGLM::ZOOM = EValue<float>(kZOOM, "1");
float      SGLM::TMIN = EValue<float>(kTMIN, "0.1");
float      SGLM::TMAX = EValue<float>(kTMAX, "100.0");
int        SGLM::CAM  = SCAM::EGet(kCAM, "perspective") ;
int        SGLM::NEARFAR = SBAS::EGet(kNEARFAR, "gazelength") ;
int        SGLM::FOCAL   = SBAS::EGet(kFOCAL,   "gazelength") ;
int        SGLM::FULLSCREEN  = EValue<int>(kFULLSCREEN,   "0") ;
int        SGLM::ESCALE  = SBAS::EGet(kESCALE,  "extent") ;  // "asis"
float      SGLM::EXTENT_FUDGE = EValue<float>(kEXTENT_FUDGE, "1");
uint32_t   SGLM::VIZMASK = SBitSet::Value<uint32_t>(32, kVIZMASK, "t" );
int        SGLM::TRACEYFLIP  = ssys::getenvint(kTRACEYFLIP,  0 ) ;
int        SGLM::LEVEL  = ssys::getenvint(kLEVEL,  0 ) ;
float      SGLM::TIMESCALE = EValue<float>(kTIMESCALE, "1.0");


float      SGLM::T0 = EValue<float>(kT0, "0.0" );
float      SGLM::T1 = EValue<float>(kT1, "0.0" );
float      SGLM::TT = EValue<float>(kTT, "0.0" );
int        SGLM::TN = ssys::getenvint(kTN, 5000 );


inline void SGLM::SetWH( int width, int height ){ WH.x = width ; WH.y = height ; }
inline void SGLM::SetCE(  float x, float y, float z, float w){ CE.x = x ; CE.y = y ; CE.z = z ;  CE.w = w ; }

inline void SGLM::SetEYE( float x, float y, float z){ EYE.x = x  ; EYE.y = y  ; EYE.z = z  ;  EYE.w = 1.f ; }
inline void SGLM::SetLOOK(float x, float y, float z){ LOOK.x = x ; LOOK.y = y ; LOOK.z = z ;  LOOK.w = 1.f ; }
inline void SGLM::SetUP(  float x, float y, float z){ UP.x = x   ; UP.y = y   ; UP.z = z   ;  UP.w = 0.f ; }  // 0.f as treat as direction

inline void SGLM::SetZOOM( float v ){ ZOOM = v ; if(LEVEL>0) std::cout << "SGLM::SetZOOM " << ZOOM << std::endl ; }
inline void SGLM::SetTMIN( float v ){ TMIN = v ; if(LEVEL>0) std::cout << "SGLM::SetTMIN " << TMIN << std::endl ; }
inline void SGLM::SetTMAX( float v ){ TMAX = v ; if(LEVEL>0) std::cout << "SGLM::SetTMAX " << TMAX << std::endl ; }

inline void SGLM::IncZOOM( float v ){ ZOOM += v ; /*std::cout << "SGLM::IncZOOM " << ZOOM << std::endl ;*/ }
inline void SGLM::IncTMIN( float v ){ TMIN += v ; /*std::cout << "SGLM::IncTMIN " << TMIN << std::endl ;*/ }
inline void SGLM::IncTMAX( float v ){ TMAX += v ; if(LEVEL>0) std::cout << "SGLM::IncTMAX " << TMAX << std::endl ; }


inline void SGLM::SetCAM( const char* cam ){ CAM = SCAM::Type(cam) ; }
inline void SGLM::SetNEARFAR( const char* nearfar ){ NEARFAR = SBAS::Type(nearfar) ; }
inline void SGLM::SetFOCAL( const char* focal ){ FOCAL = SBAS::Type(focal) ; }
inline void SGLM::SetVIZMASK( const char* vizmask ){ VIZMASK = SBitSet::Value<uint32_t>(32, vizmask) ; }
inline void SGLM::SetTRACEYFLIP( const char* traceyflip ){ TRACEYFLIP = SBAS::AsInt(traceyflip) ; }

bool SGLM::is_vizmask_set(unsigned ibit) const { return SBitSet::IsSet<uint32_t>(vizmask, ibit ); }





inline int SGLM::Width(){  return WH.x ; }
inline int SGLM::Height(){ return WH.y ; }
inline int SGLM::Width_Height(){ return Width()*Height() ; }

inline float SGLM::Aspect() { return float(WH.x)/float(WH.y) ; }
inline const char* SGLM::CAM_Label(){ return SCAM::Name(CAM) ; }
inline const char* SGLM::NEARFAR_Label(){ return SBAS::Name(NEARFAR) ; }
inline const char* SGLM::FOCAL_Label(){   return SBAS::Name(FOCAL) ; }
inline const char* SGLM::ESCALE_Label(){   return SBAS::Name(ESCALE) ; }
inline std::string SGLM::VIZMASK_Label(){   return SBitSet::DescValue(VIZMASK) ; }
inline const char* SGLM::TRACEYFLIP_Label(){  return SBAS::DescInt(TRACEYFLIP) ; }

inline void SGLM::Copy(float* dst, const glm::vec3& src )
{
    dst[0] = src.x ;
    dst[1] = src.y ;
    dst[2] = src.z ;
}
inline void SGLM::Copy(float* dst, const glm::vec4& src )
{
    dst[0] = src.x ;
    dst[1] = src.y ;
    dst[2] = src.z ;
    dst[3] = src.w ;
}





inline SGLM::SGLM()
    :
    SGLM_DESC(ssys::getenvbool(_SGLM_DESC)),
    tree(nullptr),
    scene(nullptr),
    rtp_tangential(false),
    extent_scale(false),
    model2world(1.f),
    world2model(1.f),
    initModelMatrix_branch(-1),
    eye(   0.f,0.f,0.f),
    look(  0.f,0.f,0.f),
    up(    0.f,0.f,0.f),
    gaze(  0.f,0.f,0.f),
    eye2look(1.f),
    look2eye(1.f),
    q_lookrot(1.f,0.f,0.f,0.f),   // identity quaternion
    q_eyerot( 1.f,0.f,0.f,0.f),   // identity quaternion
    eyeshift(0.f,0.f,0.f),
    depth_test(1),
    home_count(0),
    forward_ax(0.f,0.f,0.f),
    right_ax(0.f,0.f,0.f),
    top_ax(0.f,0.f,0.f),
    rot_ax(1.f),
    world2camera(1.f),
    camera2world(1.f),
    u(0.f,0.f,0.f),
    v(0.f,0.f,0.f),
    w(0.f,0.f,0.f),
    e(0.f,0.f,0.f),
    cam(CAM),
    nearfar(NEARFAR),   // gazelength default
    focal(FOCAL),
    fullscreen(FULLSCREEN),
    vizmask(VIZMASK),
    traceyflip(TRACEYFLIP),
    rendertype(0),
    nearfar_manual(0.f),
    focal_manual(0.f),

    near(0.1f),   // units of get_nearfar_basis
    far(5.f),     // units of get_nearfar_basis
    orthographic_scale(1.f),

    projection(1.f),
    zproj(0.f, 0.f, 0.f, 0.f),
    MV(1.f),
    IMV(1.f),
    MV_ptr(glm::value_ptr(MV)),
    MVP(1.f),
    MVP_ptr(glm::value_ptr(MVP)),
    IDENTITY(1.f),
    IDENTITY_ptr(glm::value_ptr(IDENTITY)),
    title("SGLM"),
    ar(nullptr),
    br(nullptr),
    gs(nullptr),
    enabled_time_bump(true),
    enabled_time_halt(false),
    timeparam_ptr(glm::value_ptr(timeparam))
{
    init();
}

inline void SGLM::init()
{
    addlog("SGLM::init", "ctor");
    INSTANCE = this ;

    axes.push_back( {1.f,0.f,0.f} );
    axes.push_back( {0.f,1.f,0.f} );
    axes.push_back( {0.f,0.f,1.f} );

    constrain();
}


inline void SGLM::setTreeScene( stree* _tree, SScene* _scene )
{
    tree = _tree ;
    scene = _scene ;

    moi_fr = tree->get_frame_moi();

    bool DUMP = ssys::getenvbool(__setTreeScene_DUMP) ;
    if(DUMP) std::cout
        << "SGLM::setTreeScene "
        << __setTreeScene_DUMP
        << " DUMP " << ( DUMP ? "YES" : "NO " )
        << " moi_fr \n"
        << moi_fr.desc()
        << "\n"
        ;

}

inline void SGLM::handle_frame_hop(int wanted_frame_idx)
{
    bool frame_hop = !has_frame_idx(wanted_frame_idx) ;

    if(SGLM_DESC) std::cout
        << "SGLM::handle_frame_hop"
        << " wanted_frame_idx " << wanted_frame_idx
        << " frame_hop " << ( frame_hop ? "YES" : "NO " )
        << "\n"
        ;

    if(frame_hop)
    {
        if( wanted_frame_idx == -2 )
        {
            if(SGLM_DESC) std::cout << _SGLM_DESC << "\n"  << desc() ;
            set_frame(moi_fr);
        }
        else if( wanted_frame_idx >= 0 )
        {
            assert(scene);  // must setTreeScene before using handle_frame_hop
            sfr wfr = scene->getFrame(wanted_frame_idx) ;
            set_frame(wfr);
        }
    }
}





void SGLM::setLookRotation(float angle_deg, glm::vec3 axis )
{
    q_lookrot = glm::angleAxis( glm::radians(angle_deg), glm::normalize(axis) );
}
void SGLM::setEyeRotation(float angle_deg, glm::vec3 axis )
{
    q_eyerot = glm::angleAxis( glm::radians(angle_deg), glm::normalize(axis) );
}


void SGLM::setDepthTest(int _depth_test)
{
    depth_test = _depth_test ;
}
int SGLM::depthTest() const
{
    return depth_test ;
}



/**
SGLM::setLookRotation
--------------------------

In "Rotate" mode, after pressing "R", as drag the mouse
around get different orientations of the look position.

**/

void SGLM::setLookRotation( const glm::vec2& a, const glm::vec2& b )
{
    //std::cout << "SGLM::setLookRotation " << glm::to_string(a) << " " << glm::to_string(b) << std::endl ;
    q_lookrot = SGLM_Arcball::A2B_Screen( a, b );
}
void SGLM::setEyeRotation( const glm::vec2& a, const glm::vec2& b )
{
    //std::cout << "SGLM::setEyeRotation " << glm::to_string(a) << " " << glm::to_string(b) << std::endl ;
    q_eyerot = SGLM_Arcball::A2B_Screen( a, b );
}



void SGLM::cursor_moved_action( const glm::vec2& a, const glm::vec2& b, unsigned modifiers )
{
    if(SGLM_Modnav::IsR(modifiers))
    {
        setLookRotation(a,b);
    }
    else if(SGLM_Modnav::IsY(modifiers))
    {
        setEyeRotation(a,b);
    }
    else
    {
        setEyeRotation(a,b);
        //key_pressed_action(modifiers);
    }
}


/**
SGLM::key_pressed_action
-------------------------

Currently only from SGLFW::key_repeated

**/

void SGLM::key_pressed_action( unsigned modifiers )
{
    float factor = SGLM_Modifiers::IsShift(modifiers) ? 5.f : 1.f ;
    float speed = factor*extent()/100. ;

    if(SGLM_Modnav::IsW(modifiers)) eyeshift.z += speed ;
    if(SGLM_Modnav::IsS(modifiers)) eyeshift.z -= speed ;

    if(SGLM_Modnav::IsA(modifiers)) eyeshift.x += speed ; // sign surprised me here
    if(SGLM_Modnav::IsD(modifiers)) eyeshift.x -= speed ;

    if(SGLM_Modnav::IsQ(modifiers)) eyeshift.y += speed ;
    if(SGLM_Modnav::IsE(modifiers)) eyeshift.y -= speed ;
}


void SGLM::home()
{
    if(LEVEL > 3) std::cout << "SGLM::home [" << home_count << "]" << descEyeShift();
    home_count += 1 ;

    eyeshift.x = 0.f ;
    eyeshift.y = 0.f ;
    eyeshift.z = 0.f ;
    q_lookrot = SGLM_Arcball::Identity();
    q_eyerot = SGLM_Arcball::Identity();

    SetZOOM(1.f);
}

std::string SGLM::descEyeShift() const
{
    std::stringstream ss ;
    ss
       << "[SGLM::descEyeShift\n"
       << " eyeshift " << Present(eyeshift) << "\n"
       << " q_lookrot " << DescQuat(q_lookrot) << "\n"
       << " q_eyerot " << DescQuat(q_eyerot) << "\n"
       << "]SGLM::descEyeShift\n"
       ;
    std::string str = ss.str() ;
    return str ;
}

std::string SGLM::DescQuat( const glm::quat& q ) // static
{
    glm::mat4 m = glm::mat4_cast(q);
    std::stringstream ss ;

    ss << "q_wxyz{"
       << " " << std::setw(10) << std::fixed << std::setprecision(3) << q.w
       << "," << std::setw(10) << std::fixed << std::setprecision(3) << q.x
       << "," << std::setw(10) << std::fixed << std::setprecision(3) << q.y
       << "," << std::setw(10) << std::fixed << std::setprecision(3) << q.z
       << "}\n"
       << m
       << "\n"
       ;

    std::string str = ss.str() ;
    return str ;
}





void SGLM::tcam()
{
    cam = SCAM::Next(cam);
}

void SGLM::toggle_traceyflip()
{
    traceyflip = !traceyflip ;
}
void SGLM::toggle_rendertype()
{
    rendertype = !rendertype ;
}

/**
SGLM::Command
--------------

**/

void SGLM::Command(const SGLM_Parse& parse, SGLM* gm, bool dump)  // static
{
    assert( parse.key.size() == parse.val.size() );
    int num_kv = parse.key.size();
    int num_op = parse.opt.size();

    for(int i=0 ; i < num_kv ; i++)
    {
        const char* k = parse.key[i].c_str();
        const char* v = parse.val[i].c_str();

        if(dump) std::cout
           << "SGLM::Command"
           << " k[" << ( k ? k : "-" ) << "]"
           << " v[" << ( v ? v : "-" ) << "]"
           << std::endl
           ;

        if(     strcmp(k,"ce")==0)
        {
            glm::vec4 tmp = SVec4(v, 0.f) ;
            SetCE( tmp.x, tmp.y, tmp.z, tmp.w );
        }
        else if(     strcmp(k,"eye")==0)
        {
            glm::vec3 tmp = SVec3(v, 0.f) ;
            SetEYE( tmp.x, tmp.y, tmp.z );
        }
        else if(strcmp(k,"look")==0)
        {
            glm::vec3 tmp = SVec3(v, 0.f) ;
            SetLOOK( tmp.x, tmp.y, tmp.z );
        }
        else if(strcmp(k,"up")==0)
        {
            glm::vec3 tmp = SVec3(v, 0.f) ;
            SetUP( tmp.x, tmp.y, tmp.z );
        }
        else if(strcmp(k,"zoom")==0)     SetZOOM(SValue<float>(v)) ;
        else if(strcmp(k,"tmin")==0)     SetTMIN(SValue<float>(v)) ;
        else if(strcmp(k,"tmax")==0)     SetTMAX(SValue<float>(v)) ;
        else if(strcmp(k,"inc-zoom")==0) IncZOOM(SValue<float>(v)) ;
        else if(strcmp(k,"inc-tmin")==0) IncTMIN(SValue<float>(v)) ;
        else if(strcmp(k,"inc-tmax")==0) IncTMAX(SValue<float>(v)) ;
        else if(strcmp(k,"inc-time")==0) gm->inc_time(SValue<float>(v)) ;
        else
        {
            std::cout << "SGLM::Command unhandled kv [" << k << "," << v << "]" << std::endl ;
        }
    }

    for(int i=0 ; i < num_op ; i++)
    {
        const char* op = parse.opt[i].c_str();
        if(     strcmp(op,"desc")==0) std::cout << gm->desc() << std::endl ;
        else if(strcmp(op,"home")==0) gm->home();
        else if(strcmp(op,"tcam")==0) gm->tcam();
        else if(strcmp(op,"traceyflip")==0) gm->toggle_traceyflip();
        else if(strcmp(op,"rendertype")==0) gm->toggle_rendertype();
        else
        {
            std::cout << "SGLM::Command IGNORING op [" << ( op ? op : "-" ) << "]" << std::endl;
        }
    }
}


/**
SGLM::command
--------------

The objective of this method is to provide a generic method
to control view parameters without requiring tight coupling between
this struct which handles view maths and various rendering systems.
For example key callbacks into SGLFW yield control strings that
can be passed here to change the view, where SGLFW need only know
the SCMD interface that this method fulfils.
Similarly UDP commands from remote commandlines picked up
by async listeners can similarly change the view.

From old opticks see::

    oglrap/OpticksViz::command
    okc/Composition::command
    okc/Camera::commandNear

**/

int SGLM::command(const char* cmd)
{
    SGLM_Parse parse(cmd);

    bool dump = false ;
    if(dump) std::cout << "SGLM::command" << std::endl << parse.desc() ;
    Command(parse, this, dump);
    update();
    return 0 ;
}


void SGLM::save(const char* dir, const char* stem) const
{
    fr.save( dir, stem ); // .npy
    writeDesc( dir, stem, ".log" );
}

void SGLM::writeDesc(const char* dir, const char* name_ , const char* ext_ ) const
{
    std::string ds = desc() ;
    const char* name = name_ ? name_ : "SGLM__writeDesc" ;
    const char* ext  = ext_ ? ext_ : ".log" ;
    NP::WriteString(dir, name, ext,  ds );
}


/**
SGLM::desc
------------

Invoke this from interactive viz using "P" key

**/

std::string SGLM::desc() const
{
    std::stringstream ss ;
    ss << descFrame() << std::endl ;
    ss << DescInput() << std::endl ;
    ss << descInput() << std::endl ;
    ss << descModelMatrix() << std::endl ;
    ss << descELU() << std::endl ;
    ss << descNearFar() << std::endl ;
    ss << descEyeSpace() << std::endl ;
    ss << descEyeBasis() << std::endl ;
    ss << descProj() << std::endl ;
    ss << descProjection() << std::endl ;
    ss << descBasis() << std::endl ;
    ss << descLog() << std::endl ;
    ss << desc_MVP() << std::endl ;
    ss << desc_MVP_ce_corners() << std::endl ;
    ss << desc_MVP_ce_midface() << std::endl ;
    std::string s = ss.str();
    return s ;
}
void SGLM::dump() const
{
    std::cout << desc() << std::endl ;
}

std::string SGLM::DescInput() // static
{
    std::stringstream ss ;
    ss << "SGLM::DescInput" << std::endl ;
    ss << std::setw(15) << "SGLM::CAM"  << " " << SGLM::CAM << std::endl ;
    ss << std::setw(15) << kCAM << " " << CAM_Label() << std::endl ;
    ss << std::setw(15) << kNEARFAR << " " << NEARFAR_Label() << std::endl ;
    ss << std::setw(15) << kFOCAL   << " " << FOCAL_Label() << std::endl ;
    ss << std::setw(15) << kESCALE  << " " << ESCALE_Label() << std::endl ;
    ss << std::setw(15) << kEXTENT_FUDGE  << " " << EXTENT_FUDGE << std::endl ;
    ss << std::setw(15) << kWH    << Present( WH )   << " Aspect " << Aspect() << std::endl ;
    ss << std::setw(15) << kCE    << Present( CE )   << std::endl ;
    ss << std::setw(15) << kEYE   << Present( EYE )  << std::endl ;
    ss << std::setw(15) << kLOOK  << Present( LOOK ) << std::endl ;
    ss << std::setw(15) << kUP    << Present( UP )   << std::endl ;
    ss << std::setw(15) << kZOOM  << Present( ZOOM ) << std::endl ;
    ss << std::endl ;
    std::string s = ss.str();
    return s ;
}

std::string SGLM::descInput() const
{
    std::stringstream ss ;
    ss << "SGLM::descInput" << std::endl ;
    ss << std::setw(25) << " sglm.fr.desc "  << fr.desc()  << std::endl ;
    ss << std::setw(25) << " sglm.cam " << cam << std::endl ;
    ss << std::setw(25) << " SCAM::Name(sglm.cam) " << SCAM::Name(cam) << std::endl ;
    std::string s = ss.str();
    return s ;
}

/**
SGLM::set_frame
-----------------

**/


inline void SGLM::set_frame( const sfr& fr_ )
{
    fr = fr_ ;
    //std::cout << "SGLM::set_frame [" << fr.get_name() << "]\n";

    update();

    int DUMP = ssys::getenvint(_DUMP, 0);
    if(DUMP > 0) std::cout << _DUMP << ":" << DUMP << "\n" << desc() ;
}

inline int SGLM::get_frame_idx() const { return fr.get_idx(); }
inline bool SGLM::has_frame_idx(int q_idx) const
{
    int curr_idx = get_frame_idx() ;
    if(false) std::cout << "SGLM::has_frame_idx" << " q_idx " << q_idx << " curr_idx " << curr_idx << "\n" ;
    return q_idx == curr_idx ;
}
inline const std::string& SGLM::get_frame_name() const { return fr.get_name(); }

/**
SGLM::extent
-------------

When looking at small objects a fudged increase in the extent with EXTENT_FUDGE improves
the viz interface by avoiding overly tight near,far and also avoiding overly slow
WASDQE navigation.

**/

inline float SGLM::extent() const {   return EXTENT_FUDGE*(fr.ce.w > 0 ? fr.ce.w : CE.w) ; }
inline float SGLM::tmin_abs() const { return extent()*TMIN ; }  // HUH:extent might not be the basis ?
inline float SGLM::tmax_abs() const { return extent()*TMAX ; }  // HUH:extent might not be the basis ?

/**
SGLM::update
--------------

initModelMatrix
    model2world, world2model from frame or ce (translation only, not including extent scale)
    [note the only? use of these is from initELU]

initELU
    eye,look,up in world frame from EYE,LOOK,UP in "ce" frame by using model2world
    and doing extent scaling


updateGaze
    eye,look -> gaze (world frame)
    eye2look,look2eye (eye frame)

updateEyeSpace
    gaze,up,eye -> world2camera

updateEyeBasis
    Transforms eye/camera basis vectors using *camera2world* matrix
    obtained from SGLM::updateEyeSpace into world frame, with
    scaling depending on Aspect, ZOOM and focal_basis to
    yield (u,v,w,e) basis vec3 that are used by CSGOptiX::prepareRenderParam
    to setup the raytrace render params.

updateNearFar
    scales extent relative inputs TMIN(eg 0.1) and TMAX(eg 100) by extent to get
    world frame Near and Far distances ... HUH: not so simple as near far are divided
    by the nearfar_basis that defaults to gazelength but can be extent

    HMM: thats non-intuitive, could explain mis-behaviour

    [recently moved this from after initELU to before updateProjection
     as ELU+EyeSpace+EyeBasis belong together as do NearFar+Projection ]

updateProjection

updateComposite
    putting together the composite transforms that OpenGL uses


**/


inline void SGLM::update()
{
    addlog("SGLM::update", "[");

    constrain();

    initModelMatrix();  //  fr.ce(center)->model2world translation
    initELU();          //  EYE,LOOK,UP,model2world,extent->eye,look,up

    updateGaze();       //  eye,look,up->gaze,eye2look,look2eye
    updateEyeSpace();   //  gaze,up,eye->world2camera,camera2world

    updateNearFar();     // TMIN,TMAX-> near,far
    updateProjection();  // focal_basis,ZOOM,aspect,near,far -> projection

    updateComposite();   // projection,word2camera -> MVP
    updateEyeBasis();   //  IMV, camera2world,apect,ZOOM,focal_basis,...->u,v,w,e  [used for ray trace rendering]

    updateTitle();

    constrain();
    addlog("SGLM::update", "]");
}

/**
SGLM::constrain
----------------

UP is a direction vector, not a position, so non-zero UP.w would be a bug

**/

inline void SGLM::constrain() const
{
    bool expect_UP_w = UP.w == 0.f ;
    if(!expect_UP_w) std::cerr
        << "SGLM::constrain"
        << " expect_UP_w " << ( expect_UP_w ? "YES" : "NO " )
        << " UP " << Present(UP)
        << descELU()
        << "\n"
        ;

    assert( expect_UP_w );
}


inline void SGLM::set_rtp_tangential(bool rtp_tangential_ )
{
    rtp_tangential = rtp_tangential_ ;
    addlog("set_rtp_tangential", rtp_tangential );
}

inline void SGLM::set_extent_scale(bool extent_scale_ )
{
    extent_scale = extent_scale_ ;
    addlog("set_extent_scale", extent_scale );
}



/**
SGLM::initModelMatrix   (formerly updateModelMatrix)
------------------------------------------------------

Because this depends on the input geometry ce it
seems more appropriate to prefix with "init" rather than "update"
Thats because changing CE is currently an initialization only thing.

Called by SGLM::update.

initModelMatrix_branch:1
    used when the sframe transforms are not identity,
    just take model2world and world2model from sframe m2w w2m

initModelMatrix_branch:2
    used for rtp_tangential:true (not default)
    TODO: this calc now done in CSGTarget::getFrameComponents
    does it need to be here too ?

initModelMatrix_branch:3
    form model2world and world2model matrices
    from fr.ce alone, ignoring the frame transforms

    For consistency with the transforms from sframe.h
    the escale is not included into model2world/world2model,
    that is done in SGLM::initELU.

    So currently initModelMatrix only handles translation from CE center,
    not extent scaling.

**/

inline void SGLM::initModelMatrix()
{
    initModelMatrix_branch = 0 ;

    // NOTE THAT THESE SPECIAL CASES ARE THE ONLY NON-CE USES OF sframe.h
    // SUGGESTS REMOVE sframe.h from SGLM passing instead normally the ce
    // and the transforms in the special case where needed

    //bool m2w_not_identity = fr.m2w.is_identity(sframe::EPSILON) == false ;
    //bool w2m_not_identity = fr.w2m.is_identity(sframe::EPSILON) == false ;

    bool fr_has_transform = !fr.is_identity() ;
    if( fr_has_transform )
    {
        initModelMatrix_branch = 1 ;
        //model2world = glm::make_mat4x4<float>(fr.m2w.cdata());
        //world2model = glm::make_mat4x4<float>(fr.w2m.cdata());

        model2world = fr.m2w ;
        world2model = fr.w2m ;
    }
    else if( rtp_tangential )
    {
        initModelMatrix_branch = 2 ;
        SCenterExtentFrame<double> cef( fr.ce.x, fr.ce.y, fr.ce.z, fr.ce.w, rtp_tangential, extent_scale );
        model2world = cef.model2world ;
        world2model = cef.world2model ;
        // HMM: these matrix might have extent scaling already ?
    }
    else
    {
        initModelMatrix_branch = 3 ;
        glm::vec3 tr(fr.ce.x, fr.ce.y, fr.ce.z) ;

        float f = 1.f ; // get_escale_() ;
        assert( f > 0.f );
        glm::vec3 sc(f, f, f) ;
        glm::vec3 isc(1.f/f, 1.f/f, 1.f/f) ;

        addlog("initModelMatrix.3.fabricate", f );

        model2world = glm::scale(glm::translate(glm::mat4(1.0), tr), sc);
        world2model = glm::translate( glm::scale(glm::mat4(1.0), isc), -tr);
    }
    addlog("initModelMatrix", initModelMatrix_branch );
}
std::string SGLM::descModelMatrix() const
{
    std::stringstream ss ;
    ss << "SGLM::descModelMatrix" << std::endl ;
    ss << " sglm.model2world \n" << Present( model2world ) << std::endl ;
    ss << " sglm.world2model \n" << Present( world2model ) << std::endl ;
    ss << " sglm.initModelMatrix_branch " << initModelMatrix_branch << std::endl ;
    ss << std::endl ;
    std::string s = ss.str();
    return s ;
}

float SGLM::get_escale_() const
{
    float escale = 0.f ;
    switch(ESCALE)
    {
        case BAS_EXTENT: escale = extent() ; break ;
        case BAS_ASIS:   escale = 1.f      ; break ;
    }
    return escale ;
}

/**
SGLM::get_escale
==================

Returns escale matrix (which typically comes from extent fr.ce.w), eg with extent of 9.0::

    9.0   0.0    0.0    0.0
    0.0   9.0    0.0    0.0
    0.0   0.0    9.0    0.0
    0.0   0.0    0.0    1.0

**/

glm::mat4 SGLM::get_escale() const
{
    float f = get_escale_();
    //std::cout << "SGLM::get_escale f " << f << "\n" ;

    glm::vec3 sc(f,f,f) ;
    glm::mat4 esc = glm::scale(glm::mat4(1.f), sc);
    return esc ;
}

/**
SGLM::initELU
-----------------

Uses escale matrix (which typically comes from extent fr.ce.w), eg with extent of 9.0
to convert the inputs (EYE, LOOK, UP) in units of extent, eg defaults::

    UP     (0,0,1,0)
    LOOK   (0,0,0,1)
    EYE    (-1,-1,0,1)
    "GAZE" (1,1,0,0)

into world frame by applying the extent with the escale matrix
to give vec3 : up,look,eye  eg::

    up     (0,0,9)
    look   (0,0,0)
    eye    (0,0,9)

The advantage of using units of extent for the view inputs
is that the view will then often provide something visible
with geometry of any size.


Q: Why not include extent scaling in the model2world matrix ?
A: This is for consistency with sframe.h transforms which are used when
   non-identity transforms are provided with the frame.

**/

void SGLM::initELU()
{
    glm::mat4 escale = get_escale();

    eye  = glm::vec3( model2world * escale * EYE ) ;
    look = glm::vec3( model2world * escale * LOOK ) ;
    up   = glm::vec3( model2world * escale * UP ) ;

    if(LEVEL > 0) std::cout
        << "[ SGLM::initELU\n"
        << descELU()
        << "] SGLM::initELU\n"
        ;
}

/**
SGLM::updateGaze
------------------

gaze
    vector from eye->look  look-eye::


               look
                +
               /
              / / gaze
             /
            +
           eye

eye2look
    transform that translates from eye to look
    (as in camera/eye frame this is along z-direction only)

look2eye
    transform that translates from look to eye
    (as in camera/eye frame this is along z-direction only)


* NB using gazelen invariance between world and eye frames
  (no scaling for those makes that valid)

* HMM: is the sign convention here correct ? (OpenGL -Z is forward)


**/

void SGLM::updateGaze()
{
    gaze = glm::vec3( look - eye ) ;
    avoidDegenerateBasisByChangingUp();

    float gazlen = getGazeLength();
    eye2look = glm::translate( glm::mat4(1.), glm::vec3(0,0,gazlen));
    look2eye = glm::translate( glm::mat4(1.), glm::vec3(0,0,-gazlen));
}

float SGLM::getGazeLength()  const { return glm::length(gaze) ; }   // must be after updateGaze
float SGLM::getGazeCrossUp() const { return glm::length(glm::cross(glm::normalize(gaze), up))  ; }


void SGLM::avoidDegenerateBasisByChangingUp()
{
    float eps = 1e-4f ;
    float gcu = getGazeCrossUp() ;
    if(gcu > eps) return ;

    for(int i=0 ; i < 3 ; i++)
    {
        up = axes[i] ;
        gcu = getGazeCrossUp() ;
        if(gcu > eps)
        {
            std::cout
                << "SGLM::avoidDegenerateBasisByChangingUp"
                << " gaze " << glm::to_string(gaze)
                << " up " << glm::to_string(up)
                << " GazeCrossUp " << gcu
                << " ( avoid this message by starting with more sensible EYE,LOOK,UP basis ) "
                << std::endl
                ;
            return ;
        }
    }
    assert( gcu > eps );
}




std::string SGLM::descELU() const
{
    float escale_ = get_escale_();
    glm::mat4 escale = get_escale();
    std::stringstream ss ;
    ss << "[SGLM::descELU\n"
       << " [" << kLEVEL << "] " << LEVEL   << "\n"
       << " EYE  "  << Present( EYE )       << "\n"
       << " LOOK "  << Present( LOOK )      << "\n"
       << " UP   "  << Present( UP )        << "\n"
       << " GAZE "  << Present( LOOK-EYE )  << "\n"
       << "\n"
       << " escale_ " << Present( escale_ ) << "\n"
       << " escale\n" << Present( escale )  << "\n"
       << " model2world\n" << Present( model2world ) << "\n"
       << " (model2world * escale)\n"   << Present( model2world * escale ) << "\n"
       << "\n"
       << " EYE*escale  "  << Present( EYE*escale )  << "\n"
       << " LOOK*escale "  << Present( LOOK*escale ) << "\n"
       << " UP*escale   "  << Present( UP*escale )   << "\n"
       << " GAZE*escale "  << Present( (LOOK-EYE)*escale ) << "\n"
       << "\n"
       << " eye  = (model2world * escale * EYE  ) "  << Present( model2world * escale * EYE ) << "\n"
       << " look = (model2world * escale * LOOK ) "  << Present( model2world * escale * LOOK ) << "\n"
       << " up   = (model2world * escale * UP   ) "  << Present( model2world * escale * UP  ) << "\n"
       << " gaze                                  "  << Present( gaze ) << "\n"
       << "]SGLM::descELU\n"
       ;
    std::string str = ss.str();
    return str ;
}




/**
SGLM::updateEyeSpace
---------------------

NB "eye" and "camera" are used interchangeably, meaning the same thing

Form world2camera camera2world from eye position and
gaze and up directions in world frame together with
OpenGL convention.

Normalized eye space oriented via world frame gaze and up directions.

        +Y    -Z
     top_ax  forward_ax    (from normalized gaze vector)
         |  /
         | /
         |/
         +----- right_ax  (+X)
        .
       .
      .
    -forward_ax
    +Z


world2camera
    transforms a world frame coordinate into a camera(aka eye) frame coordinate
    the transform is formed from first a translation to the origin
    of the "eye" world frame coordinate followed by a rotation following
    the OpenGL eye space convention : -Z is forward, +X to right, +Y up

**/

void SGLM::updateEyeSpace() // gaze,up,eye -> world2camera
{
    forward_ax = glm::normalize(gaze);  // gaze is from eye->look "look - eye"
    right_ax   = glm::normalize(glm::cross(forward_ax,up));
    top_ax     = glm::normalize(glm::cross(right_ax,forward_ax));

    // OpenGL eye space convention : -Z is forward, +X to right, +Y up
    rot_ax[0] = glm::vec4( right_ax, 0.f );
    rot_ax[1] = glm::vec4( top_ax  , 0.f );
    rot_ax[2] = glm::vec4( -forward_ax, 0.f );
    rot_ax[3] = glm::vec4( 0.f, 0.f, 0.f, 1.f );

    glm::mat4 ti(glm::translate(glm::vec3(eye)));  // origin to eye
    glm::mat4 t(glm::translate(glm::vec3(-eye)));  // eye to origin

    world2camera = glm::transpose(rot_ax) * t  ;
    camera2world = ti * rot_ax ;
}

std::string SGLM::descEyeSpace() const
{
    std::stringstream ss ;
    ss << "SGLM::descEyeSpace" << std::endl ;
    ss << std::setw(15) << "sglm.forward_ax" << Present(forward_ax) << std::endl ;
    ss << std::setw(15) << "sglm.right_ax"   << Present(right_ax) << std::endl ;
    ss << std::setw(15) << "sglm.top_ax"     << Present(top_ax) << std::endl ;
    ss << std::endl ;

    ss << " sglm.world2camera \n" << Present( world2camera ) << std::endl ;
    ss << " sglm.camera2world \n" << Present( camera2world ) << std::endl ;
    ss << std::endl ;

    std::string s = ss.str();
    return s ;
}




/**
SGLM::updateEyeBasis
----------------------

Transforms eye/camera basis vectors using *camera2world* matrix
obtained from SGLM::updateEyeSpace into world frame, with
scaling depending on Aspect, ZOOM and focal_basis to
yield (u,v,w,e) basis vec3 that are used by CSGOptiX::prepareRenderParam
to setup the raytrace render params.

Note how are using inverted transforms for the ray tracing
basis compared to the rasterization ones. Also no projection
matrix for ray tracing as thats inherent in the technique.

Getting this to feel the look rotation quaternion
was done by changing from using the camera2world
matrix to use the IVM InverseModelView
thats calculated in updateComposite.  As a result
the order of the update calculation was changed
moving this to after updateComposite.

::

    Y:top
       |  .-Z:gaz
       | .
       |.
       +----- X:rht
      /
    +Z


**/

void SGLM::updateEyeBasis()
{
    // eye basis vectors using OpenGL convention
    glm::vec4 rht( 1., 0., 0., 0.);  // +X
    glm::vec4 top( 0., 1., 0., 0.);  // +Y
    glm::vec4 gaz( 0., 0.,-1., 0.);  // -Z

    // eye position in eye frame
    glm::vec4 ori( 0., 0., 0., 1.);

    float aspect = Aspect() ;
    //float fsc = get_focal_basis() ;   // default is gazelength
    float fsc = get_transverse_scale() ;

    float fscz = fsc/ZOOM  ;          // increased ZOOM decreases field-of-view
    //float lsc = getGazeLength() ;
    float lsc = get_near_abs() ;

    u = glm::vec3( IMV * rht ) * fscz * aspect ;
    v = glm::vec3( IMV * top ) * fscz  ;
    w = glm::vec3( IMV * gaz ) * lsc ;
    e = glm::vec3( IMV * ori );

    wnorm = glm::normalize(w);

}




/**
SGLM::updateNearFar
--------------------------

scales extent relative inputs TMIN(eg 0.1) and TMAX(eg 100) by extent to get
world frame Near and Far distances

HUH: but then tmi,tmx get divided by nearfar basis, which could be extent
(default is gazelength). Thats confusing.

As the basis needs gazelength, this must be done after updateELU
but isnt there still a problem of basis consistency between tmin_abs and set_near_abs ?
For example extent scaling vs gazelength scaling ?

**/
void SGLM::updateNearFar()
{
    float tmi = tmin_abs() ;
    addlog("updateNearFar.tmi", tmi);
    set_near_abs(tmi) ;

    float tmx = tmax_abs() ;
    addlog("updateNearFar.tmx", tmx);
    set_far_abs(tmx) ;

}
std::string SGLM::descNearFar() const
{
    std::stringstream ss ;
    ss << "SGLM::descNearFar" << std::endl ;
    std::string s = ss.str();
    return s ;
}


/**
SGLM::updateTitle
------------------

The *title* is set as the cxr_min.sh OpenGL window title by SGLFW::renderloop_tail

**/


void SGLM::updateTitle()
{
    std::stringstream ss ;
    ss
       << fr.get_name()
       << " "
       << fr.desc_ce()
       << " sglm.e(c2w*ori) [" << Present(e) << "] SGLM::updateTitle"
       ;
    title = ss.str();
}


void SGLM::left_right_bottom_top_near_far(lrbtnf& p) const
{
    //float fsc = get_focal_basis() ;
    float fsc = get_transverse_scale() ;
    float fscz = fsc/ZOOM  ;
    float aspect = Aspect();

    p.left   = -aspect*fscz ;
    p.right  =  aspect*fscz ;
    p.bottom = -fscz ;
    p.top    =  fscz ;
    p.near   =  get_near_abs() ;
    p.far    =  get_far_abs()  ;
}


/**
SGLM::updateProjection
-----------------------

Suspect that for consistency of rasterized and ray traced
renders this will need to match SGLM::updateEyeBasis better in
the z-direction.

glm::frustum


**/

void SGLM::updateProjection()
{
    left_right_bottom_top_near_far(proj);
    assert( cam == CAM_PERSPECTIVE || cam == CAM_ORTHOGRAPHIC );
    switch(cam)
    {
       case CAM_PERSPECTIVE:  projection = glm::frustum( proj.left, proj.right, proj.bottom, proj.top, proj.near, proj.far ) ; break ;
       case CAM_ORTHOGRAPHIC: projection = glm::ortho(   proj.left, proj.right, proj.bottom, proj.top, proj.near, proj.far ) ; break ;
    }

    FillZProjection(zproj, projection);



    if(ssys::getenvbool(_updateProjection_DEBUG))
    {
        std::cout
            << _updateProjection_DEBUG
            << " zproj("
            << std::setw(10) << std::fixed << std::setprecision(4) << zproj.x << " "
            << std::setw(10) << std::fixed << std::setprecision(4) << zproj.y << " "
            << std::setw(10) << std::fixed << std::setprecision(4) << zproj.z << " "
            << std::setw(10) << std::fixed << std::setprecision(4) << zproj.w << " "
            << ")"
            << "\n"
            ;

        if(cam == CAM_PERSPECTIVE) std::cout
            << " proj.A_frustum "
            << std::setw(10) << std::fixed << std::setprecision(4) << proj.A_frustum()
            << " proj.B_frustum "
            << std::setw(10) << std::fixed << std::setprecision(4) << proj.B_frustum()
            << "\n"
            ;

        if(cam == CAM_ORTHOGRAPHIC) std::cout
            << " proj.A_ortho "
            << std::setw(10) << std::fixed << std::setprecision(4) << proj.A_ortho()
            << " proj.B_ortho "
            << std::setw(10) << std::fixed << std::setprecision(4) << proj.B_ortho()
            << "\n"
            ;
     }

}

/**
SGLM::FillZProjection
-----------------------

After the ancient okc Camera::fillZProjection

See ~/o/notes/issues/impl_composited_rendering_in_7plus_workflow.rst

Opticks conventional matrix memory layout has
translation in slots [12 13 14]

          0   1   2    3
          4   5   6    7
          8   9  10   11
        [12  13  14]  15

glm::mat4 element addressing is (row, col) (see stra_test::Elements)
so this is grabbing the third column.

          0  1 | 2|  3
          4  5 | 6|  7
          8  9 |10| 11
         12 13 |14| 15

Looking at glm/ext/matrix_clip_space.inl::

    159     template<typename T>
    160     GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> frustumRH_NO(T left, T right, T bottom, T top, T nearVal, T farVal)
    161     {
    162         mat<4, 4, T, defaultp> Result(0);
    163         Result[0][0] = (static_cast<T>(2) * nearVal) / (right - left);
    164         Result[1][1] = (static_cast<T>(2) * nearVal) / (top - bottom);
    165         Result[2][0] = (right + left) / (right - left);
    166         Result[2][1] = (top + bottom) / (top - bottom);
    167         Result[2][2] = - (farVal + nearVal) / (farVal - nearVal);
    168         Result[2][3] = static_cast<T>(-1);
    169         Result[3][2] = - (static_cast<T>(2) * farVal * nearVal) / (farVal - nearVal);
    170         return Result;
    171     }

Copying from above frustumRH_NO and expressing in glm::mat4 memory element order
as standard for Opticks (NB transposed representation is more commonly shown)::


   |     2n/(r-l)          0       {    0          }   0   |
   |                               {               }       |
   |         0        2n/(t-b)     {    0          }   0   |
   |                               {               }       |
   |     (r+l)/(r-l)  (t+b)/(t-b)  { -(f+n)/(f-n)  }  -1   |
   |                               {               }       |
   |         0            0        { -2.*f*n/(f-n) }   0   |


For perspective projection this grabs::

   { 0 ,   0,  -(f+n)/(f-n),  -2.*f*n/(f-n) }


Looking at glm/ext/matrix_clip_space.inl::

     54     template<typename T>
     55     GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> orthoRH_NO(T left, T right, T bottom, T top, T zNear, T zFar)
     56     {
     57         mat<4, 4, T, defaultp> Result(1);
     58         Result[0][0] = static_cast<T>(2) / (right - left);
     59         Result[1][1] = static_cast<T>(2) / (top - bottom);
     60         Result[2][2] = - static_cast<T>(2) / (zFar - zNear);
     61         Result[3][0] = - (right + left) / (right - left);
     62         Result[3][1] = - (top + bottom) / (top - bottom);
     63         Result[3][2] = - (zFar + zNear) / (zFar - zNear);
     64         return Result;
     65     }

Copying from above orthoRH_NO and expressing in glm::mat4 memory element order
as standard for Opticks (NB transposed representation is more commonly shown)::


    |   2/(r-l)        0              0             0   |
    |     0           2/(t-b)         0             0   |
    |     0            0            -2/(f-n)        0   |
    |  -(r+l)/(r-l)   -(t+b)/(t-b)  -(f+n)/(f-n)    1   |


For ortho this grabs::

    { 0,   0,   -2/(f-n),   -(f+n)/(f-n)  }


**/


void SGLM::FillZProjection(glm::vec4& _ZProj, const glm::mat4& _Proj) // static
{
    _ZProj.x = _Proj[0][2] ;
    _ZProj.y = _Proj[1][2] ;
    _ZProj.z = _Proj[2][2] ;
    _ZProj.w = _Proj[3][2] ;
}

/**
SGLM::FillAltProjection
----------------------

For perspective projection this grabs::

   {   (r+l)/(r-l) , (t+b)/(t-b) , -(f+n)/(f-n) ,  -1 }


**/

void SGLM::FillAltProjection(glm::vec4& _AProj, const glm::mat4& _Proj) // static
{
    _AProj.x = _Proj[2][0] ;
    _AProj.y = _Proj[2][1] ;
    _AProj.z = _Proj[2][2] ;
    _AProj.w = _Proj[2][3] ;
}


inline float lrbtnf::A_frustum() const { return -(far+near)/(far-near) ; }
inline float lrbtnf::B_frustum() const { return -2.f*far*near/(far-near) ; }
inline float lrbtnf::A_ortho() const   { return -2.f/(far-near) ; }
inline float lrbtnf::B_ortho() const   { return -(far+near)/(far-near) ; }

inline std::string lrbtnf::desc() const
{
    std::stringstream ss ;
    ss << "lrbtnf::desc (inputs to glm::frustum OR glm::ortho )"
       << " l " << std::setw(7) << std::fixed << std::setprecision(3) << left
       << " r " << std::setw(7) << std::fixed << std::setprecision(3) << right
       << " b " << std::setw(7) << std::fixed << std::setprecision(3) << bottom
       << " t " << std::setw(7) << std::fixed << std::setprecision(3) << top
       << " n " << std::setw(7) << std::fixed << std::setprecision(3) << near
       << " f " << std::setw(7) << std::fixed << std::setprecision(3) << far
       << " A_frustum:-(f+n)(f-n) " << std::setw(7) << std::fixed << std::setprecision(3) << A_frustum()
       << " B_frustum:-2fn/(f-n)  " << std::setw(7) << std::fixed << std::setprecision(3) << B_frustum()
       << " A_ortho:-2/(f-n)      " << std::setw(7) << std::fixed << std::setprecision(3) << A_ortho()
       << " B_ortho:-(f+n)/(f-n)  " << std::setw(7) << std::fixed << std::setprecision(3) << B_ortho()
       << "\n"
       ;
    std::string str = ss.str();
    return str ;
}

/**
SGLM::zdepth_pos
-------------------

http://www.songho.ca/opengl/gl_projectionmatrix.html

**/


float SGLM::zdepth_pos( const glm::tvec4<float>& p_eye ) const
{
    glm::tvec4<float> c = projection * p_eye ;
    float zd = c.z/c.w ;
    return zd ;
}

/**
SGLM::zdepth0
--------------


**/


float SGLM::zdepth0( const float& z_eye ) const
{
    const float& ze = z_eye ;
    float A = zproj_A() ;
    float B = zproj_B() ;
    float zd(0.f) ;

    assert( cam == CAM_PERSPECTIVE || cam == CAM_ORTHOGRAPHIC );
    switch(cam)
    {
        case CAM_PERSPECTIVE:  zd = -(A + B/ze) ; break ;    //zd = (A*ze + B)/(-ze) ;
        case CAM_ORTHOGRAPHIC: zd = A*ze + B    ; break ;
    }
    return zd ;
}

/**
SGLM::zdepth1
--------------

Compare with CSGOptiX7.cu:render::

    235     float eye_z = -prd->distance()*dot(params.WNORM, direction) ;
    236     const float& A = params.ZPROJ.z ;
    237     const float& B = params.ZPROJ.w ;
    238     float zdepth = cameratype == 0u ? -(A + B/eye_z) : A*eye_z + B  ;  // cf SGLM::zdepth1

**/
float SGLM::zdepth1( const float& z_eye ) const
{
    const float& ze = z_eye ;
    const float& A = zproj.z ;
    const float& B = zproj.w ;
    assert( cam == CAM_PERSPECTIVE || cam == CAM_ORTHOGRAPHIC );
    return cam == CAM_PERSPECTIVE ? -(A + B/ze) : A*ze + B  ;
}

float SGLM::zproj_A() const
{
    assert( cam == CAM_PERSPECTIVE || cam == CAM_ORTHOGRAPHIC );
    float A(0.f);
    switch(cam)
    {
        case CAM_PERSPECTIVE:  A = proj.A_frustum() ; break ;
        case CAM_ORTHOGRAPHIC: A = proj.A_ortho()   ; break ;
    }
    return A ;
}

float SGLM::zproj_B() const
{
    assert( cam == CAM_PERSPECTIVE || cam == CAM_ORTHOGRAPHIC );
    float B(0.f);
    switch(cam)
    {
        case CAM_PERSPECTIVE:  B = proj.B_frustum() ; break ;
        case CAM_ORTHOGRAPHIC: B = proj.B_ortho()   ; break ;
    }
    return B ;
}







float SGLM::get_transverse_scale() const
{
    assert( cam == CAM_PERSPECTIVE || cam == CAM_ORTHOGRAPHIC );
    //return cam == CAM_ORTHOGRAPHIC ? orthographic_scale : get_near_abs() ;
    return get_near_abs() ;
}


/**
SGLM::updateComposite
----------------------

Putting together the composite transforms that OpenGL needs

* contrast with old Opticks ~/o/optickscore/Composition.cc Composition::update

* note the conjugte of a quaternion rotation represents the inverse rotation

**/

void SGLM::updateComposite()
{
    //std::cout << "SGLM::updateComposite" << std::endl ;

    glm::mat4 _eyeshift = glm::translate(glm::mat4(1.0), eyeshift ) ;    // eyeshift starts (0,0,0) changed by WASDQE keys
    glm::mat4 _ieyeshift = glm::translate(glm::mat4(1.0), -eyeshift ) ;

    glm::mat4 _lookrot = glm::mat4_cast(q_lookrot) ;
    glm::mat4 _ilookrot = glm::mat4_cast( glm::conjugate(q_lookrot) ) ;

    glm::mat4 _eyerot = glm::mat4_cast(q_eyerot) ;
    glm::mat4 _ieyerot = glm::mat4_cast( glm::conjugate( q_eyerot )) ;


    MV = _eyeshift * _eyerot * look2eye * _lookrot * eye2look * world2camera ;  // just world2camera before shifts, rotations

    IMV = camera2world * look2eye * _ilookrot * eye2look * _ieyerot  * _ieyeshift  ;
    //IMV = glm::inverse( MV );

    MVP = projection * MV ;    // MVP aka world2clip (needed by OpenGL shader pipeline)
}



std::string SGLM::descEyeBasis() const
{
    std::stringstream ss ;
    ss << "SGLM::descEyeBasis : camera frame basis vectors transformed into world and focal plane scaled " << std::endl ;

    int wid = 25 ;
    float aspect = Aspect() ;
    float fsc = get_focal_basis() ;
    float fscz = fsc/ZOOM ;
    float gazlen = getGazeLength() ;

    ss << std::setw(wid) << "aspect" << Present(aspect) << std::endl ;
    ss << std::setw(wid) << "near " << Present(near) << std::endl ;
    ss << std::setw(wid) << "ZOOM " << Present(ZOOM) << std::endl ;
    ss << std::setw(wid) << "get_focal_basis"      << Present(fsc) << std::endl ;
    ss << std::setw(wid) << "get_focal_basis/ZOOM" << Present(fscz) << std::endl ;
    ss << std::setw(wid) << "getGazeLength " << Present(gazlen) << std::endl ;

    ss << std::setw(wid) << "sglm.e " << Present(e) << " glm::vec3( camera2world * ori ) " << std::endl ;
    ss << std::setw(wid) << "sglm.u " << Present(u) << " glm::vec3( camera2world * rht ) * fsc * aspect " << std::endl ;
    ss << std::setw(wid) << "sglm.v " << Present(v) << " glm::vec3( camera2world * top ) * fsc  " << std::endl ;
    ss << std::setw(wid) << "sglm.w " << Present(w) << " glm::vec3( camera2world * gaz ) * gazlen  " << std::endl ;
    std::string s = ss.str();
    return s ;
}

std::string SGLM::DescEyeBasis( const glm::vec3& E, const glm::vec3& U, const glm::vec3& V, const glm::vec3& W )
{
    int wid = 15 ;
    std::stringstream ss ;
    ss << "SGLM::DescEyeBasis E,U,V,W " << std::endl ;
    ss << std::setw(wid) << "E " << Present(E) << std::endl ;
    ss << std::setw(wid) << "U " << Present(U) << std::endl ;
    ss << std::setw(wid) << "V " << Present(V) << std::endl ;
    ss << std::setw(wid) << "W " << Present(W) << std::endl ;
    std::string s = ss.str();
    return s ;
}










void SGLM::set_nearfar_mode(const char* mode){    addlog("set_nearfar_mode",  mode) ; nearfar = SBAS::Type(mode) ; }
void SGLM::set_focal_mode(  const char* mode){    addlog("set_focal_mode",    mode) ; focal   = SBAS::Type(mode) ; }

const char* SGLM::get_nearfar_mode() const { return SBAS::Name(nearfar) ; }
const char* SGLM::get_focal_mode() const {   return SBAS::Name(focal) ; }

void SGLM::set_nearfar_manual(float nearfar_manual_ ){ addlog("set_nearfar_manual", nearfar_manual_ ) ; nearfar_manual = nearfar_manual_  ; }
void SGLM::set_focal_manual(float focal_manual_ ){     addlog("set_focal_manual", focal_manual_ )     ; focal_manual = focal_manual_  ; }

/**
SGLM::get_focal_basis
----------------------

BAS_NEARABS problematic as

**/

float SGLM::get_focal_basis() const
{
    float basis = 0.f ;
    switch(focal)
    {
        case BAS_MANUAL:     basis = focal_manual        ; break ;
        case BAS_EXTENT:     basis = fr.ce.w             ; break ;
        case BAS_GAZELENGTH: basis = getGazeLength()     ; break ;  // only available after updateELU
        case BAS_NEARABS:    basis = get_near_abs()      ; break ;
    }
    return basis ;
}



void SGLM::set_near( float near_ ){ near = near_ ; addlog("set_near", near); }
void SGLM::set_far(  float far_ ){  far = far_   ; addlog("set_far", far);   }
float SGLM::get_near() const  { return near ; }
float SGLM::get_far()  const  { return far  ; }

/**
SGLM::get_nearfar_basis
-------------------------

Default is gazelength



**/

float SGLM::get_nearfar_basis() const
{
    float basis = 0.f ;
    switch(nearfar)
    {
        case BAS_MANUAL:     basis = nearfar_manual      ; break ;
        case BAS_EXTENT:     basis = extent()            ; break ;  // only after set_frame
        case BAS_GAZELENGTH: basis = getGazeLength()     ; break ;  // only available after updateELU (default)
        case BAS_NEARABS:    assert(0)                   ; break ;  // this mode only valud for get_focal_basis (as near far in units of this)
    }
    return basis ;
}


// CAUTION: depends on get_nearfar_basis
void SGLM::set_near_abs( float near_abs_ )
{
    float nfb = get_nearfar_basis() ;
    float nab = near_abs_/nfb ;
    addlog("set_near_abs.arg", near_abs_) ;
    addlog("set_near_abs.nfb", nfb );
    addlog("set_near_abs.nab", nab );
    set_near( nab ) ;
}
void SGLM::set_far_abs(  float far_abs_ )
{
    float nfb = get_nearfar_basis() ;
    float fab = far_abs_/nfb ;
    addlog("set_far_abs.arg", far_abs_)   ;
    set_far( fab ) ;
}

/**
SGLM::get_near_abs
--------------------

Used from CSGOptiX::prepareRenderParam for tmin

OptixLaunch with negative tmin throws exception  OPTIX_EXCEPTION_CODE_INVALID_RAY

**/
float SGLM::get_near_abs() const { return std::max(0.f, near*get_nearfar_basis()) ; }

/**
SGLM::get_far_abs
--------------------

Used from CSGOptiX::prepareRenderParam for tmax

**/
float SGLM::get_far_abs() const { return   far*get_nearfar_basis() ; }


std::string SGLM::descFrame() const
{
    return fr.desc();
}

std::string SGLM::descBasis() const
{
    int wid = 25 ;
    std::stringstream ss ;
    ss << "SGLM::descBasis" << std::endl ;
    ss << std::setw(wid) << " sglm.get_nearfar_mode " << get_nearfar_mode()  << std::endl ;
    ss << std::setw(wid) << " sglm.nearfar_manual "   << Present( nearfar_manual ) << std::endl ;
    ss << std::setw(wid) << " sglm.fr.ce.w  "     << Present( fr.ce.w )  << std::endl ;
    ss << std::setw(wid) << " sglm.getGazeLength  " << Present( getGazeLength() ) << std::endl ;
    ss << std::setw(wid) << " sglm.get_nearfar_basis " << Present( get_nearfar_basis() ) << std::endl ;
    ss << std::endl ;
    ss << std::setw(wid) << " sglm.near  "     << Present( near )  << " (units of nearfar basis) " << std::endl ;
    ss << std::setw(wid) << " sglm.far   "     << Present( far )   << " (units of nearfar basis) " << std::endl ;
    ss << std::setw(wid) << " sglm.get_near_abs  " << Present( get_near_abs() ) << " near*get_nearfar_basis() " << std::endl ;
    ss << std::setw(wid) << " sglm.get_far_abs  " << Present( get_far_abs() )   << " far*get_nearfar_basis() " << std::endl ;
    ss << std::endl ;
    ss << std::setw(wid) << " sglm.get_focal_mode " << get_focal_mode()  << std::endl ;
    ss << std::setw(wid) << " sglm.get_focal_basis " << Present( get_focal_basis() ) << std::endl ;
    ss << std::endl ;
    std::string s = ss.str();
    return s ;
}

std::string SGLM::descProj() const
{
    int wid = 25 ;
    std::stringstream ss ;
    ss << "SGLM::descProj" << std::endl ;
    ss << std::setw(wid) << " (lrbtnf)proj.desc " << proj.desc() << "\n" ;
    std::string str = ss.str();
    return str ;
}

std::string SGLM::descProjection() const
{
    float fsc = get_focal_basis() ;
    float fscz = fsc/ZOOM  ;
    float aspect = Aspect();
    float left   = -aspect*fscz ;
    float right  =  aspect*fscz ;
    float bottom = -fscz ;
    float top    =  fscz ;
    float near_abs   = get_near_abs() ;
    float far_abs    = get_far_abs()  ;

    int wid = 25 ;
    std::stringstream ss ;
    ss << "SGLM::descProjection" << std::endl ;
    ss << std::setw(wid) << "Aspect" << Present(aspect) << std::endl ;
    ss << std::setw(wid) << "get_focal_basis" << Present(fsc) << std::endl ;
    ss << std::setw(wid) << "get_focal_basis/ZOOM" << Present(fscz) << std::endl ;
    ss << std::setw(wid) << "ZOOM" << Present(ZOOM) << std::endl ;
    ss << std::setw(wid) << "left"   << Present(left) << std::endl ;
    ss << std::setw(wid) << "right"  << Present(right) << std::endl ;
    ss << std::setw(wid) << "top"    << Present(top) << std::endl ;
    ss << std::setw(wid) << "bottom" << Present(bottom) << std::endl ;
    ss << std::setw(wid) << "get_near_abs" << Present(near_abs) << std::endl ;
    ss << std::setw(wid) << "get_far_abs" << Present(far_abs) << std::endl ;

    ss << std::setw(wid) << "near" << Present(near) << std::endl ;
    ss << std::setw(wid) << "far"  << Present(far) << std::endl ;
    ss << std::setw(wid) << "sglm.projection\n" << Present(projection) << std::endl ;
    ss << descProj() << std::endl ;

    std::string str = ss.str();
    return str ;
}


std::string SGLM::descComposite() const
{
    int wid = 25 ;
    std::stringstream ss ;
    ss << "SGLM::descComposite" << std::endl ;
    ss << std::setw(wid) << "sglm.MVP\n" << Present(MVP) << std::endl ;
    std::string str = ss.str();
    return str ;
}






template<typename T>
void SGLM::ce_corners_world( std::vector<glm::tvec4<T>>& v_world ) const
{
    std::vector<glm::tvec4<T>> corners ;
    SCE::Corners<T>( corners, fr.ce );
    assert(corners.size() == 8 );

    for(int i=0 ; i < 8 ; i++ )
    {
        const glm::tvec4<T>& corner = corners[i];
        v_world.push_back(corner);
    }
}

template<typename T>
void SGLM::ce_midface_world( std::vector<glm::tvec4<T>>& v_world ) const
{
    std::vector<glm::tvec4<T>> midface ;
    SCE::Midface( midface, fr.ce );
    assert(midface.size() == 6+1 );

    for(int i=0 ; i < 6+1 ; i++ )
    {
        v_world.push_back(midface[i]);
    }
}



template<typename T>
void SGLM::Apply_XF( std::vector<glm::tvec4<float>>& v_out, const std::vector<glm::tvec4<T>>& v_in, const glm::tmat4x4<float>& XF, bool flip )  // static
{
    int num = v_in.size();
    for(int i=0 ; i < num ; i++ )
    {
        const glm::tvec4<float> in = v_in[i] ;  // not by ref, to allow changing type
        glm::tvec4<float> ou = flip ? XF*in : in*XF ;
        v_out.push_back(ou);
    }
}


template<typename T>
void SGLM::apply_MV( std::vector<glm::tvec4<float>>& v_eye, const std::vector<glm::tvec4<T>>& v_world, bool flip ) const
{
    Apply_XF(v_eye, v_world, MV, flip);
}
template<typename T>
void SGLM::apply_MVP( std::vector<glm::tvec4<float>>& v_clip, const std::vector<glm::tvec4<T>>& v_world, bool flip ) const
{
    Apply_XF(v_clip, v_world, MVP, flip);
}
template<typename T>
void SGLM::apply_P( std::vector<glm::tvec4<float>>& v_clip, const std::vector<glm::tvec4<T>>& v_eye, bool flip ) const
{
    const glm::tmat4x4<float>& P = projection ;
    Apply_XF(v_clip, v_eye, P, flip);
}


std::string SGLM::desc_MVP() const
{
    std::stringstream ss ;
    ss << "SGLM::desc_MVP" << std::endl ;
    ss << " MVP " << std::endl  << Present(MVP) << std::endl ;
    ss << " MVP_ptr " << std::endl  << Present<float>(MVP_ptr,16) << std::endl ;
    std::string str = ss.str();
    return str ;
}

/**
SGLM::desc_MVP_ce_corners
---------------------------------
**/

std::string SGLM::desc_MVP_ce_corners() const
{
    static const int NUM = 8 ;

    std::vector<glm::tvec4<double>> v_world ;
    ce_corners_world<double>(v_world);
    assert( v_world.size() == NUM );

    std::vector<glm::tvec4<float>> v_clip ;
    bool flip = true ;
    apply_MVP( v_clip, v_world, flip );
    assert( v_clip.size() == NUM );

    std::stringstream ss ;
    ss << "SGLM::desc_MVP_ce_corners (clipped in {})" << std::endl ;
    for(int i=0 ; i < NUM ; i++ )
    {
        const glm::tvec4<double>& _world = v_world[i] ;
        const glm::tvec4<float>& _clip = v_clip[i] ;
        glm::vec4 _ndc(_clip.x/_clip.w, _clip.y/_clip.w, _clip.z/_clip.w, 1.f );
        // normalized device coordinates : from division by clip.w

        bool clipped = IsClipped(_ndc) ;
        char bef = clipped ? '{' : ' ' ;
        char aft = clipped ? '}' : ' ' ;

        ss
            << "[" << i << "]"
            << " world " << stra<double>::Desc(_world)
            << " clip  " << stra<float>::Desc(_clip)
            << " ndc " << bef << stra<float>::Desc(_ndc) << aft
            << std::endl
            ;
    }
    std::string str = ss.str();
    return str ;
}

/**
SGLM::desc_MV_P_MVP_ce_corners
--------------------------------

Used as testing ground for the zdepth calc, see:

* notes/issues/impl_composited_rendering_in_7plus_workflow.rst
* http://www.songho.ca/opengl/gl_projectionmatrix.html

**/

std::string SGLM::desc_MV_P_MVP_ce_corners() const
{
    std::vector<glm::tvec4<float>> v_world ;
    std::vector<glm::tvec4<float>> v_eye ;
    std::vector<glm::tvec4<float>> v_clip_0 ;
    std::vector<glm::tvec4<float>> v_clip_1 ;

    static const int NUM = 8 ;
    ce_corners_world(v_world);
    assert( v_world.size() == NUM );

    bool flip = true ;
    apply_MV(   v_eye   ,  v_world, flip );
    apply_P(    v_clip_0,  v_eye  , flip );
    apply_MVP(  v_clip_1,  v_world, flip );

    assert( v_eye.size() == NUM );
    assert( v_clip_0.size() == NUM );
    assert( v_clip_1.size() == NUM );

    int wid = 25 ;

    std::stringstream ss ;
    ss << "SGLM::desc_MV_P_MVP_ce_corners" << std::endl ;

    const glm::tvec3<float>& ray_origin = eye ;
    ss << std::setw(wid) << " ray_origin    " << ' '   << stra<float>::Desc(ray_origin) << ' ' << "\n" ;
    ss << std::setw(wid) << " forward_ax    " << ' '   << stra<float>::Desc(forward_ax) << ' ' << "\n" ;
    ss << std::setw(wid) << " wnorm(front)  " << ' '   << stra<float>::Desc(wnorm)      << ' ' << "\n" ;

    for(int i=0 ; i < NUM ; i++ )
    {
        const glm::tvec4<float>& _world  = v_world[i] ;
        glm::tvec3<float> world(_world);

        const glm::tvec4<float>& _eye    = v_eye[i] ;
        const glm::tvec4<float>& _clip_0 = v_clip_0[i] ;
        const glm::tvec4<float>& _clip_1 = v_clip_1[i] ;

        // imagine ray tracing intersects at each of the corners
        glm::tvec3<float> ray_direction = glm::normalize( world - ray_origin  );
        float distance = glm::length( world - ray_origin );
        float ray_eye_z = -distance*glm::dot(forward_ax, ray_direction) ;

        float zd_p = zdepth_pos(_eye) ;
        float zd_0 = zdepth0(ray_eye_z) ;
        float zd_1 = zdepth1(ray_eye_z) ;

        glm::vec4 _ndc_0(_clip_0.x/_clip_0.w, _clip_0.y/_clip_0.w, _clip_0.z/_clip_0.w, 1.f );
        glm::vec4 _ndc_1(_clip_1.x/_clip_1.w, _clip_1.y/_clip_1.w, _clip_1.z/_clip_1.w, 1.f );
        // normalized device coordinates : from division by clip_0.w

        bool clipped_0 = IsClipped(_ndc_0) ;
        char bef_0 = clipped_0 ? '{' : ' ' ;
        char aft_0 = clipped_0 ? '}' : ' ' ;

        bool clipped_1 = IsClipped(_ndc_1) ;
        char bef_1 = clipped_1 ? '{' : ' ' ;
        char aft_1 = clipped_1 ? '}' : ' ' ;

        ss
            << "[" << i << "]\n"
            << std::setw(wid) << " ray_origin    " << ' '   << stra<float>::Desc(ray_origin) << ' ' << "\n"
            << std::setw(wid) << " _world        " << ' '   << stra<float>::Desc(_world)    << ' ' << "\n"
            << std::setw(wid) << " ray_direction " << ' '   << stra<float>::Desc(ray_direction) << ' ' << "\n"
            << std::setw(wid) << " _eye          " << ' '   << stra<float>::Desc(_eye)      << ' ' << "\n"
            << std::setw(wid) << " _clip_0       " << ' '   << stra<float>::Desc(_clip_0)   << ' ' << "\n"
            << std::setw(wid) << " _clip_1       " << ' '   << stra<float>::Desc(_clip_1)   << ' ' << "\n"
            << std::setw(wid) << " _ndc_0        " << bef_0 << stra<float>::Desc(_ndc_0)    << aft_0 << "\n"
            << std::setw(wid) << " _ndc_1        " << bef_1 << stra<float>::Desc(_ndc_1)    << aft_1 << "\n"
            << std::setw(wid) << " zd_p          " << ' '   << stra<float>::Desc(zd_p)      << ' ' << "\n"
            << std::setw(wid) << " zd_0          " << ' '   << stra<float>::Desc(zd_0)      << ' ' << "\n"
            << std::setw(wid) << " zd_1          " << ' '   << stra<float>::Desc(zd_1)      << ' ' << "\n"
            << std::setw(wid) << " ray_eye_z     " << ' '   << stra<float>::Desc(ray_eye_z) << ' ' << "\n"
            << std::endl
            ;
    }
    std::string str = ss.str();
    return str ;
}


std::string SGLM::desc_MVP_ce_midface() const
{
    static const int NUM = 6+1 ;

    std::vector<glm::tvec4<double>> v_world ;
    ce_midface_world(v_world);
    assert( v_world.size() == NUM );

    std::vector<glm::tvec4<float>> v_clip ;
    bool flip = true ;
    apply_MVP( v_clip, v_world, flip );
    assert( v_clip.size() == NUM );

    std::stringstream ss ;
    ss << "SGLM::desc_MVP_ce_midface (clipped in {})" << std::endl ;
    ss << " MVP " << std::endl  << Present(MVP) << std::endl ;
    for(int i=0 ; i < NUM ; i++ )
    {
        const glm::tvec4<double>& _world = v_world[i] ;
        const glm::tvec4<float>& _clip = v_clip[i] ;
        glm::vec4 _ndc(_clip.x/_clip.w, _clip.y/_clip.w, _clip.z/_clip.w, 1.f );
        // normalized device coordinates : from division by clip.w

        bool clipped = IsClipped(_ndc) ;
        char bef = clipped ? '[' : ' ' ;
        char aft = clipped ? ']' : ' ' ;

        ss
            << "[" << i << "]"
            << " world " << stra<double>::Desc(_world)
            << " clip  " << stra<float>::Desc(_clip)
            << " ndc " << bef << Present(_ndc) << aft
            << std::endl
            ;

    }
    std::string str = ss.str();
    return str ;
}


bool SGLM::IsClipped(const glm::vec4& _ndc ) // static
{
    return _ndc.x < -1.f || _ndc.y < -1.f || _ndc.z < -1.f
        || _ndc.x >  1.f || _ndc.y >  1.f || _ndc.z >  1.f ;
}









inline void SGLM::addlog( const char* label, float value )
{
    std::stringstream ss ;
    ss << std::setw(25) << label << " : " << std::setw(10) << std::fixed << std::setprecision(3) << value ;
    std::string s = ss.str();
    log.push_back(s);
}

inline void SGLM::addlog( const char* label, const char* value )
{
    std::stringstream ss ;
    ss << std::setw(25) << label << " : " << value ;
    std::string s = ss.str();
    log.push_back(s);
}

inline std::string SGLM::descLog() const
{
    std::stringstream ss ;
    ss << "SGLM::descLog" << std::endl ;
    for(unsigned i=0 ; i < log.size() ; i++) ss << log[i] << std::endl ;
    std::string s = ss.str();
    return s ;
}



template <typename T>
inline T SGLM::ato_( const char* a )   // static
{
    std::string s(a);
    std::istringstream iss(s);
    T v ;
    iss >> v ;
    return v ;
}

template float    SGLM::ato_<float>( const char* a );
template unsigned SGLM::ato_<unsigned>( const char* a );
template int      SGLM::ato_<int>( const char* a );


template <typename T>
inline void SGLM::Str2Vector( std::vector<T>& vec, const char* uval ) // static
{
    std::stringstream ss(uval);
    std::string s ;
    while(getline(ss, s, ',')) vec.push_back(ato_<T>(s.c_str()));
}

template void SGLM::Str2Vector(std::vector<float>& vec,    const char* uval );
template void SGLM::Str2Vector(std::vector<int>& vec,      const char* uval );
template void SGLM::Str2Vector(std::vector<unsigned>& vec, const char* uval );


template <typename T>
inline void SGLM::GetEVector(std::vector<T>& vec, const char* key, const char* fallback )  // static
{
    const char* sval = getenv(key);
    const char* uval = sval ? sval : fallback ;
    Str2Vector(vec, uval);
}

template void SGLM::GetEVector(std::vector<float>& vec, const char* key, const char* fallback ) ;
template void SGLM::GetEVector(std::vector<int>& vec, const char* key, const char* fallback ) ;
template void SGLM::GetEVector(std::vector<unsigned>& vec, const char* key, const char* fallback ) ;



template <typename T>
inline std::string SGLM::Present(std::vector<T>& vec) // static
{
    std::stringstream ss ;
    for(unsigned i=0 ; i < vec.size() ; i++) ss << vec[i] << " " ;
    return ss.str();
}

template<typename T>
inline std::string SGLM::Present(const T* tt, int num) // static
{
    std::stringstream ss ;
    for(int i=0 ; i < num ; i++)
        ss
            << ( i % 4 == 0 && num > 4 ? ".\n" : "" )
            << " " << std::fixed << std::setw(10) << std::setprecision(4) << tt[i]
            << ( i == num-1 && num > 4 ? ".\n" : "" )
            ;

    std::string str = ss.str();
    return str ;
}






inline void SGLM::GetEVec(glm::vec3& v, const char* key, const char* fallback ) // static
{
    std::vector<float> vec ;
    SGLM::GetEVector<float>(vec, key, fallback);
    std::cout << key << " " << Present(vec) << std::endl ;
    assert( vec.size() == 3 );
    for(int i=0 ; i < 3 ; i++) v[i] = vec[i] ;
}

inline void SGLM::GetEVec(glm::vec4& v, const char* key, const char* fallback ) // static
{
    std::vector<float> vec ;
    SGLM::GetEVector<float>(vec, key, fallback);
    std::cout << key << " " << Present(vec) << std::endl ;
    assert( vec.size() == 4 );
    for(int i=0 ; i < 4 ; i++) v[i] = vec[i] ;
}

inline glm::vec4 SGLM::EVec4(const char* key, const char* fallback, float missing) // static
{
    std::vector<float> vec ;
    SGLM::GetEVector<float>(vec, key, fallback);
    glm::vec4 v ;
    for(int i=0 ; i < 4 ; i++) v[i] = i < int(vec.size()) ? vec[i] : missing   ;
    return v ;
}
inline glm::vec4 SGLM::SVec4(const char* str, float missing) // static
{
    std::vector<float> vec ;
    SGLM::Str2Vector<float>(vec, str);
    glm::vec4 v ;
    for(int i=0 ; i < 4 ; i++) v[i] = i < int(vec.size()) ? vec[i] : missing   ;
    return v ;
}
inline glm::vec3 SGLM::SVec3(const char* str, float missing) // static
{
    std::vector<float> vec ;
    SGLM::Str2Vector<float>(vec, str);
    glm::vec3 v ;
    for(int i=0 ; i < 3 ; i++) v[i] = i < int(vec.size()) ? vec[i] : missing   ;
    return v ;
}




template <typename T>
inline T SGLM::SValue( const char* uval )  // static
{
    std::string s(uval);
    T value = ato_<T>(s.c_str());
    return value ;
}
template <typename T>
inline T SGLM::EValue( const char* key, const char* fallback )  // static
{
    const char* sval = getenv(key);
    const char* uval = sval ? sval : fallback ;
    return SValue<T>(uval);
}




inline glm::ivec2 SGLM::EVec2i(const char* key, const char* fallback ) // static
{
    std::vector<int> vec ;
    SGLM::GetEVector<int>(vec, key, fallback);
    glm::ivec2 v ;
    for(int i=0 ; i < 2 ; i++) v[i] = i < int(vec.size()) ? vec[i] : 0  ;
    return v ;
}

inline glm::vec3 SGLM::EVec3(const char* key, const char* fallback ) // static
{
    std::vector<float> vec ;
    SGLM::GetEVector<float>(vec, key, fallback);
    glm::vec3 v ;
    for(int i=0 ; i < 3 ; i++) v[i] = i < int(vec.size()) ? vec[i] : 0.f  ;
    return v ;
}
inline std::string SGLM::Present(const glm::ivec2& v, int wid )
{
    std::stringstream ss ;
    ss << std::setw(wid) << v.x << " " ;
    ss << std::setw(wid) << v.y << " " ;
    std::string s = ss.str();
    return s;
}

inline std::string SGLM::Present(const float v, int wid, int prec)
{
    std::stringstream ss ;
    ss << std::fixed << std::setw(wid) << std::setprecision(prec) << v ;
    std::string s = ss.str();
    return s;
}


inline std::string SGLM::Present(const glm::vec2& v, int wid, int prec)
{
    std::stringstream ss ;
    ss << std::fixed << std::setw(wid) << std::setprecision(prec) << v.x << " " ;
    ss << std::fixed << std::setw(wid) << std::setprecision(prec) << v.y << " " ;
    std::string s = ss.str();
    return s;
}

inline std::string SGLM::Present(const glm::vec3& v, int wid, int prec)
{
    std::stringstream ss ;
    ss << std::fixed << std::setw(wid) << std::setprecision(prec) << v.x << " " ;
    ss << std::fixed << std::setw(wid) << std::setprecision(prec) << v.y << " " ;
    ss << std::fixed << std::setw(wid) << std::setprecision(prec) << v.z << " " ;
    std::string s = ss.str();
    return s;
}


inline std::string SGLM::Present(const glm::vec4& v, int wid, int prec)
{
    std::stringstream ss ;
    ss << std::fixed << std::setw(wid) << std::setprecision(prec) << v.x << " " ;
    ss << std::fixed << std::setw(wid) << std::setprecision(prec) << v.y << " " ;
    ss << std::fixed << std::setw(wid) << std::setprecision(prec) << v.z << " " ;
    ss << std::fixed << std::setw(wid) << std::setprecision(prec) << v.w << " " ;
    std::string s = ss.str();
    return s;
}

inline std::string SGLM::Present(const float4& v, int wid, int prec)
{
    std::stringstream ss ;
    ss << std::fixed << std::setw(wid) << std::setprecision(prec) << v.x << " " ;
    ss << std::fixed << std::setw(wid) << std::setprecision(prec) << v.y << " " ;
    ss << std::fixed << std::setw(wid) << std::setprecision(prec) << v.z << " " ;
    ss << std::fixed << std::setw(wid) << std::setprecision(prec) << v.w << " " ;
    std::string s = ss.str();
    return s;
}





inline std::string SGLM::Present(const glm::mat4& m, int wid, int prec)
{
    std::stringstream ss ;
    for (int j=0; j<4; j++)
    {
        for (int i=0; i<4; i++) ss << std::fixed << std::setprecision(prec) << std::setw(wid) << m[i][j] ;
        ss << std::endl ;
    }
    return ss.str();
}


template<typename T>
inline std::string SGLM::Present_(const glm::tmat4x4<T>& m, int wid, int prec)
{
    std::stringstream ss ;
    for (int j=0; j<4; j++)
    {
        for (int i=0; i<4; i++) ss << std::fixed << std::setprecision(prec) << std::setw(wid) << m[i][j] ;
        ss << std::endl ;
    }
    return ss.str();
}


template<typename T>
inline glm::tmat4x4<T> SGLM::DemoMatrix(T scale)  // static
{
    std::array<T, 16> demo = {{
        T(1.)*scale,   T(2.)*scale,   T(3.)*scale,   T(4.)*scale ,
        T(5.)*scale,   T(6.)*scale,   T(7.)*scale,   T(8.)*scale ,
        T(9.)*scale,   T(10.)*scale,  T(11.)*scale,  T(12.)*scale ,
        T(13.)*scale,  T(14.)*scale,  T(15.)*scale,  T(16.)*scale
      }} ;
    return glm::make_mat4x4<T>(demo.data()) ;
}


/**
SGLM::setRecord
--------------------

Invoked for example from renderers like cxr_min.sh to enable
rendering of record step points together with geometry.

**/


inline void SGLM::setRecord( SRecord* _ar, SRecord* _br )
{
    ar = _ar ;
    br = _br ;

    init_time();
}

inline void SGLM::setGenstep( SGen* _gs )
{
    gs = _gs ;
}



/**
SGLM::init_time
----------------------

t0
    start time
t1
    end time

tn:int
    number of render calls with which to increment
    the event time from t0 to t1 : typically a large
    value like 5000 to avoid excessively fast animation


timeparam quad:

   +---------+----------+--------------+-------------+
   | t_start | t_stop   |  t_step      | t_current   |
   +=========+==========+==============+=============+
   |  t0     |   t1     |  (t1-t0)/tn  |     t       |
   +---------+----------+--------------+-------------+

**/


inline void SGLM::init_time()
{
    float t0 = T0 ;
    float t1 = T1 ;
    int tn = TN ;

    bool t_noenv = t0 == 0.f && t1 == 0.f ;  // envvars T0 and T1 unset or non-sensical
    if(t_noenv)
    {
        if( ar && br )
        {
            t0 = std::min( ar->get_t0(), br->get_t0() );
            t1 = std::max( ar->get_t1(), br->get_t1() );
        }
        else if( ar )
        {
            t0 = ar->get_t0();
            t1 = ar->get_t1();
        }
        else if( br )
        {
            t0 = br->get_t0();
            t1 = br->get_t1();
        }
    }

    assert( tn > 1 );

    timeparam.x = t0 ;
    timeparam.y = t1 ;
    timeparam.z = (t1-t0)/float(tn) ;
    timeparam.w = t0 ;

    bool init_time_DUMP = ssys::getenvbool(__init_time_DUMP) ;

    if(init_time_DUMP) std::cout
         << "SGLM::init_time"
         << " [" << __init_time_DUMP << "] "
         << " " << ( init_time_DUMP ? "YES" : "NO " )
         << "\n"
         << " t_noenv " << ( t_noenv ? "YES" : "NO " )
         << "\n"
         << " ar\n "
         << ( ar ? ar->desc() : "-" )
         << "\n"
         << " br\n "
         << ( br ? br->desc() : "-" )
         << "\n"
         << desc_time()
         << "\n"
         ;

}

inline void SGLM::reset_time()
{
    float t0 = get_t0();
    set_time(t0) ;
}

/**
SGLM::reset_time_TT
--------------------

SGLFW.h invokes this from renderloop when press shift+T
causing time to be reset to value of TT envvar (default 0.f)
and the animation to be disabled.

Resume animation with alt+T

**/

inline void SGLM::reset_time_TT()
{
    std::cout << "SGLM::reset_time_TT " << TT << "\n" ;
    set_time(TT) ;
}

inline void SGLM::toggle_time_halt()
{
    enabled_time_halt = !enabled_time_halt ;
}


std::string SGLM::desc_time() const
{
    float t = get_time();
    float t0 = get_t0();
    float t1 = get_t1();
    float ts = get_ts();
    int   tn = get_tn();

    std::stringstream ss ;
    ss
       << "[SGLM::desc_time\n"
       << " T0 " << std::fixed << std::setw(7) << std::setprecision(3) << T0 << "\n"
       << " T1 " << std::fixed << std::setw(7) << std::setprecision(3) << T1 << "\n"
       << " t  " << std::fixed << std::setw(7) << std::setprecision(3) << t  << "\n"
       << " t0 " << std::fixed << std::setw(7) << std::setprecision(3) << t0 << "\n"
       << " t1 " << std::fixed << std::setw(7) << std::setprecision(3) << t1 << "\n"
       << " ts " << std::fixed << std::setw(7) << std::setprecision(3) << ts << "\n"
       << " tn "  <<  std::setw(6) << tn << "\n"
       << "]SGLM::desc_time\n"
       ;

    std::string str = ss.str();
    return str ;

}



inline float SGLM::get_t0() const
{
    return timeparam.x ;
}
inline float SGLM::get_t1() const
{
    return timeparam.y ;
}
inline float SGLM::get_ts() const
{
    return timeparam.z ;
}
inline int SGLM::get_tn() const
{
    float t0 = get_t0();
    float t1 = get_t1();
    float ts = get_ts();
    return int((t1 - t0)/ts ) ;
}



inline float SGLM::get_time() const
{
    return timeparam.w ;
}
inline bool SGLM::in_timerange(float t) const
{
    return t >= timeparam.x && t <= timeparam.y ;
}



/**
SGLM::set_time
----------------

Whem t is not within the t0 to t1 range sets time to t0

**/

inline void SGLM::set_time( float t )
{
    timeparam.w = in_timerange(t) ? t : timeparam.x ;
}


inline void SGLM::time_bump()
{
    if(!enabled_time_bump || enabled_time_halt) return ;
    set_time( get_time() + get_ts() );
}

inline void SGLM::inc_time(float dy)
{
    std::cout
        << "SGLM::inc_time"
        << " dy " << dy
        << " TIMESCALE " << TIMESCALE
        << " dy*TIMESCALE " << dy*TIMESCALE
        << "\n"
        ;

    set_time( get_time() + dy*TIMESCALE );
}



inline void SGLM::renderloop_head()
{


}


/**
SGLM::renderloop_tail
----------------------

Invoked from SGLFW::renderloop_tail

At each call the simulation time is bumped until the
time exceeds t1 at which point it is returned to t0.

**/

inline void SGLM::renderloop_tail()
{
    if( option.A || option.B ) time_bump();
}

