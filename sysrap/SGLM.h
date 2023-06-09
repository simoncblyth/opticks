#pragma once
/**
SGLM : Header Only Viz Math giving ray tracing basis and rasterization projection transforms
===============================================================================================

Critical use for ray trace rendering in CSGOptiX::prepareRenderParam

SGLM.h is a single header that is replacing a boatload of classes 
used by old OpticksCore okc : Composition, View, Camera, ...
Using this aims to enable CSGOptiX to drop dependency
on okc, npy, brap and instead depend only on QUDARap, SysRap, CSG. 

* Note that animated interpolation between views available 
  in the old machinery is not yet implemented in the new workflow
  as that currently focusses on snap renders. 

* because of this incomplete nature have to keep SGLM reliance (not usage) 
  behind WITH_SGLM and retain the old Composition based version of CSGOptiX (and okc dependency) 

* note that the with Composition version of CSGOptiX also uses SGLM 
  to allow easy comparison of frames and matrices between the two approaches

* ALSO this needs to be tested by comparing rasterized and ray-traced renders
  in order to achieve full consistency. Usage with interactive changing of camera and view etc.. 
  and interactive flipping between rasterized and ray traced is the way consistency 
  of the projective and ray traced maths was achieved for okc/Composition. 

Normal inputs WH, CE, EYE, LOOK, UP are held in static variables with envvar defaults 
These can be changed with static methods before instanciating SGLM. 
NB it will usually be too late for setenv in code to influence SGLM 
as the static initialization would have happened already 
 
* https://learnopengl.com/Getting-started/Camera

TODO: WASD camera navigation, using a method intended to be called from the GLFW key callback 

TODO: provide persistency into ~16 quad4 for debugging view/cam/projection state 
      would be best to group that for clarity 


* hmm probably using nested structs makes sense, or just use SGLM namespace with 
* https://riptutorial.com/cplusplus/example/11914/nested-classes-structures
* https://www.geeksforgeeks.org/nested-structure-in-c-with-examples/

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
#include "sframe.h"
#include "NP.hh"

#include "SYSRAP_API_EXPORT.hh"


struct SYSRAP_API SGLM
{
    static SGLM* INSTANCE ; 
    static constexpr const char* kWH = "WH" ; 
    static constexpr const char* kCE = "CE" ; 
    static constexpr const char* kEYE = "EYE" ; 
    static constexpr const char* kLOOK = "LOOK" ; 
    static constexpr const char* kUP = "UP" ; 
    static constexpr const char* kZOOM = "ZOOM" ; 
    static constexpr const char* kTMIN = "TMIN" ; 
    static constexpr const char* kCAM = "CAM" ; 
    static constexpr const char* kNEARFAR = "NEARFAR" ; 
    static constexpr const char* kFOCAL = "FOCAL" ; 
    static constexpr const char* kESCALE = "ESCALE" ; 

    // static defaults, some can be overridden in the instance 
    static glm::ivec2 WH ; 

    static glm::vec4 CE ; // WHY CE? CE IS GEOMETRY NOT VIEW RELATED : PROBABLY FOR WITHOUT GEOM TEST ?
    static glm::vec4 EYE ; 
    static glm::vec4 LOOK ; 
    static glm::vec4 UP ; 

    static float ZOOM ; 
    static float TMIN ; 
    static int   CAM ; 
    static int   NEARFAR ; 
    static int   FOCAL ; 
    static int   ESCALE ; 

    static void SetWH( int width, int height ); 
    static void SetCE( float x, float y, float z, float extent ); 
    static void SetEYE( float x, float y, float z ); 
    static void SetLOOK( float x, float y, float z ); 
    static void SetUP( float x, float y, float z ); 
    static void SetZOOM( float v ); 
    static void SetTMIN( float v ); 
    static void SetCAM( const char* cam ); 
    static void SetNEARFAR( const char* nearfar ); 
    static void SetFOCAL( const char* focal ); 
    static void SetESCALE( const char* escale ); 

    // querying of static defaults 
    static std::string DescInput() ; 
    static int Width() ; 
    static int Height() ; 
    static float Aspect(); 
    static const char* CAM_Label() ; 
    static const char* NEARFAR_Label() ; 
    static const char* FOCAL_Label() ; 
    static const char* ESCALE_Label() ; 


    // member methods
    std::string descInput() const ; 

    SGLM(); 

    sframe fr ;  // CAUTION: SEvt also holds an SFrame for input photon targetting 
    void set_frame( const sframe& fr ); 
    const char* get_frame_name() const ; 
    float extent() const ; 
    float tmin_abs() const ; 

    bool rtp_tangential ; 
    void set_rtp_tangential( bool rtp_tangential_ ); 
 
    // matrices taken from fr or derived from ce when fr only holds identity
    glm::mat4 model2world ; 
    glm::mat4 world2model ; 
    int updateModelMatrix_branch ; 

    void updateModelMatrix();  // depends on ce, unless non-identity m2w and w2m matrices provided in frame
    std::string descModelMatrix() const ; 

    // world frame View converted from static model frame
    glm::vec3 eye ;  
    glm::vec3 look ; 
    glm::vec3 up ; 
    glm::vec3 gaze ; 

    float     get_escale_() const ; 
    glm::mat4 get_escale() const ; 
    void updateELU();   // depends on CE and EYE, LOOK, UP 
    float getGazeLength() const ; 
    std::string descELU() const ; 

    void updateNear(); 
    std::string descNear() const ; 

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
    glm::vec3 e ; 
    void updateEyeBasis(); 
    static std::string DescEyeBasis( const glm::vec3& E, const glm::vec3& U, const glm::vec3& V, const glm::vec3& W ); 
    std::string descEyeBasis() const ; 







    // modes
    int  cam ; 
    int  nearfar ;
    int  focal ;  

    float nearfar_manual ; 
    float focal_manual ; 
    float near ; 
    float far ; 


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


    glm::mat4 projection ; 
    glm::mat4 world2clip ; 



    void updateProjection(); 
    std::string descProjection() const ; 


    void set_nearfar_mode(const char* mode); 
    void set_focal_mode(const char* mode); 

    const char* get_nearfar_mode() const ; 
    const char* get_focal_mode() const ; 

    void set_nearfar_manual(float nearfar_manual_); 
    void set_focal_manual(float focal_manual_); 

    float get_nearfar_basis() const ; 
    float get_focal_basis() const ; 


    void writeDesc(const char* dir, const char* name="SGLM__writeDesc", const char* ext=".log" ) const ; 
    std::string desc() const ; 


    void dump() const ; 
    void update(); 
    void addlog( const char* label, float value       ) ; 
    void addlog( const char* label, const char* value ) ; 
    std::string descLog() const ; 


    template <typename T> static T ato_( const char* a );
    template <typename T> static void GetEVector(std::vector<T>& vec, const char* key, const char* fallback );
    template <typename T> static std::string Present(std::vector<T>& vec);

    static std::string Present(const glm::ivec2& v, int wid=6 );
    static std::string Present(const float v, int wid=10, int prec=3);
    static std::string Present(const glm::vec3& v, int wid=10, int prec=3);
    static std::string Present(const glm::vec4& v, int wid=10, int prec=3);
    static std::string Present(const float4& v,    int wid=10, int prec=3);
    static std::string Present(const glm::mat4& m, int wid=10, int prec=3);

    template<typename T> static std::string Present_(const glm::tmat4x4<T>& m, int wid=10, int prec=3);

    static void GetEVec(glm::vec3& v, const char* key, const char* fallback );
    static void GetEVec(glm::vec4& v, const char* key, const char* fallback );

    template <typename T> static T EValue(const char* key, const char* fallback );
    static glm::ivec2 EVec2i(const char* key, const char* fallback); 
    static glm::vec3 EVec3(const char* key, const char* fallback); 
    static glm::vec4 EVec4(const char* key, const char* fallback, float missing=1.f ); 

    template<typename T> static glm::tmat4x4<T> DemoMatrix(T scale); 
};

SGLM* SGLM::INSTANCE = nullptr ; 

glm::ivec2 SGLM::WH = EVec2i(kWH,"1920,1080") ; 
glm::vec4  SGLM::CE = EVec4(kCE,"0,0,0,100") ; 
glm::vec4  SGLM::EYE  = EVec4(kEYE, "-1,-1,0,1") ; 
glm::vec4  SGLM::LOOK = EVec4(kLOOK, "0,0,0,1") ; 
glm::vec4  SGLM::UP  =  EVec4(kUP,   "0,0,1,0") ; 
float      SGLM::ZOOM = EValue<float>(kZOOM, "1"); 
float      SGLM::TMIN = EValue<float>(kTMIN, "0.1"); 
int        SGLM::CAM  = SCAM::EGet(kCAM, "perspective") ; 
int        SGLM::NEARFAR = SBAS::EGet(kNEARFAR, "gazelength") ; 
int        SGLM::FOCAL   = SBAS::EGet(kFOCAL,   "gazelength") ; 
int        SGLM::ESCALE  = SBAS::EGet(kESCALE,  "asis") ; 

inline void SGLM::SetWH( int width, int height ){ WH.x = width ; WH.y = height ; }
inline void SGLM::SetCE(  float x, float y, float z, float w){ CE.x = x ; CE.y = y ; CE.z = z ;  CE.w = w ; }
inline void SGLM::SetEYE( float x, float y, float z){ EYE.x = x  ; EYE.y = y  ; EYE.z = z  ;  EYE.w = 1.f ; }
inline void SGLM::SetLOOK(float x, float y, float z){ LOOK.x = x ; LOOK.y = y ; LOOK.z = z ;  LOOK.w = 1.f ; }
inline void SGLM::SetUP(  float x, float y, float z){ UP.x = x   ; UP.y = y   ; UP.z = z   ;  UP.w = 1.f ; }
inline void SGLM::SetZOOM( float v ){ ZOOM = v ; }
inline void SGLM::SetTMIN( float v ){ TMIN = v ; }
inline void SGLM::SetCAM( const char* cam ){ CAM = SCAM::Type(cam) ; }
inline void SGLM::SetNEARFAR( const char* nearfar ){ NEARFAR = SBAS::Type(nearfar) ; }
inline void SGLM::SetFOCAL( const char* focal ){ FOCAL = SBAS::Type(focal) ; }

inline int SGLM::Width(){  return WH.x ; }
inline int SGLM::Height(){ return WH.y ; }
inline float SGLM::Aspect() { return float(WH.x)/float(WH.y) ; } 
inline const char* SGLM::CAM_Label(){ return SCAM::Name(CAM) ; }
inline const char* SGLM::NEARFAR_Label(){ return SBAS::Name(NEARFAR) ; }
inline const char* SGLM::FOCAL_Label(){   return SBAS::Name(FOCAL) ; }
inline const char* SGLM::ESCALE_Label(){   return SBAS::Name(ESCALE) ; }

inline SGLM::SGLM() 
    :
    rtp_tangential(false),
    model2world(1.f), 
    world2model(1.f),
    updateModelMatrix_branch(-1), 
    eye(   0.f,0.f,0.f),
    look(  0.f,0.f,0.f),
    up(    0.f,0.f,0.f),
    gaze(  0.f,0.f,0.f),
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
    nearfar(NEARFAR),
    focal(FOCAL),
    nearfar_manual(0.f),
    focal_manual(0.f),

    near(0.1f),   // units of get_nearfar_basis
    far(5.f),     // units of get_nearfar_basis

    projection(1.f),
    world2clip(1.f)
{
    addlog("SGLM::SGLM", "ctor"); 
    INSTANCE = this ; 
}


void SGLM::writeDesc(const char* dir, const char* name_ , const char* ext_ ) const 
{
    std::string ds = desc() ; 
    const char* name = name_ ? name_ : "SGLM__writeDesc" ; 
    const char* ext  = ext_ ? ext_ : ".log" ; 
    NP::WriteString(dir, name, ext,  ds );      
}

std::string SGLM::desc() const 
{
    std::stringstream ss ; 
    ss << descFrame() << std::endl ; 
    ss << DescInput() << std::endl ; 
    ss << descInput() << std::endl ; 
    ss << descModelMatrix() << std::endl ; 
    ss << descELU() << std::endl ; 
    ss << descNear() << std::endl ; 
    ss << descEyeSpace() << std::endl ; 
    ss << descEyeBasis() << std::endl ; 
    ss << descProjection() << std::endl ; 
    ss << descBasis() << std::endl ; 
    ss << descLog() << std::endl ; 
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
    ss << std::setw(25) << " sglm.fr.ce "  << Present( fr.ce )   << std::endl ; 
    ss << std::setw(25) << " sglm.cam " << cam << std::endl ; 
    ss << std::setw(25) << " SCAM::Name(sglm.cam) " << SCAM::Name(cam) << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
}

/**
SGLM::set_frame
-----------------

Avoided former kludge double call of update by repositioning updateNear 
according to its dependency on gazelength. 

**/

inline void SGLM::set_frame( const sframe& fr_ )
{
    fr = fr_ ; 
    update(); 
}

inline const char* SGLM::get_frame_name() const { return fr.get_name(); }
inline float SGLM::extent() const {   return fr.ce.w ; }
inline float SGLM::tmin_abs() const { return extent()*TMIN ; }  // HUH:extent might not be the basis ?


inline void SGLM::update()  
{
    addlog("SGLM::update", "["); 
    updateModelMatrix(); 
    updateELU(); 
    updateNear(); 
    updateEyeSpace(); 
    updateEyeBasis(); 
    updateProjection(); 
    addlog("SGLM::update", "]"); 
}

inline void SGLM::set_rtp_tangential(bool rtp_tangential_ )
{
    rtp_tangential = rtp_tangential_ ; 
    addlog("set_rtp_tangential", rtp_tangential );
}

/**
SGLM::updateModelMatrix
------------------------

Called by SGLM::update. 

updateModelMatrix_branch:1
    used when the transforms are not identity, 
    just take model2world and world2model from sframe m2w w2m  

updateModelMatrix_branch:2
    used for rtp_tangential:true (not default) 
    TODO: this calc now done in CSGTarget::getFrameComponents 
    does it need to be here too ?

updateModelMatrix_branch:3
    cookup matrix from fr.ce alone, ignoring the frame transforms 

**/

inline void SGLM::updateModelMatrix()
{
    updateModelMatrix_branch = 0 ; 

    bool m2w_not_identity = fr.m2w.is_identity(sframe::EPSILON) == false ;
    bool w2m_not_identity = fr.w2m.is_identity(sframe::EPSILON) == false ;

    if( m2w_not_identity && w2m_not_identity )
    {
        updateModelMatrix_branch = 1 ; 
        model2world = glm::make_mat4x4<float>(fr.m2w.cdata());
        world2model = glm::make_mat4x4<float>(fr.w2m.cdata());
    }
    else if( rtp_tangential )
    {
        updateModelMatrix_branch = 2 ; 
        SCenterExtentFrame<double> cef( fr.ce.x, fr.ce.y, fr.ce.z, fr.ce.w, rtp_tangential );
        model2world = cef.model2world ;
        world2model = cef.world2model ;
        // HMM: these matrix might have extent scaling already ? 
    }
    else
    {
        updateModelMatrix_branch = 3 ; 
        glm::vec3 tr(fr.ce.x, fr.ce.y, fr.ce.z) ;  

        float f = 1.f ; // get_escale_() ; 
        assert( f > 0.f ); 
        glm::vec3 sc(f, f, f) ; 
        glm::vec3 isc(1.f/f, 1.f/f, 1.f/f) ; 
        // for consistency with the transforms from sframe.h 
        // need to not include the escale here : its done below in 
        // SGLM::updateELU

        addlog("updateModelMatrix.3.fabricate", f );

        model2world = glm::scale(glm::translate(glm::mat4(1.0), tr), sc);
        world2model = glm::translate( glm::scale(glm::mat4(1.0), isc), -tr); 
    }
    addlog("updateModelMatrix", updateModelMatrix_branch );
}






std::string SGLM::descModelMatrix() const 
{
    std::stringstream ss ; 
    ss << "SGLM::descModelMatrix" << std::endl ; 
    ss << " sglm.model2world \n" << Present( model2world ) << std::endl ; 
    ss << " sglm.world2model \n" << Present( world2model ) << std::endl ; 
    ss << " sglm.updateModelMatrix_branch " << updateModelMatrix_branch << std::endl ; 
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

glm::mat4 SGLM::get_escale() const 
{
    float f = get_escale_(); 
    glm::vec3 sc(f,f,f) ; 
    glm::mat4 esc = glm::scale(glm::mat4(1.0), sc);
    return esc ; 
}


void SGLM::updateELU() // eye, look, up, gaze from model frame (extent units?) into world frame 
{
    glm::mat4 escale = get_escale(); 
    eye  = glm::vec3( model2world * escale * EYE ) ; 
    look = glm::vec3( model2world * escale * LOOK ) ; 
    up   = glm::vec3( model2world * escale * UP ) ; 
    gaze = glm::vec3( model2world * escale * (LOOK - EYE) ) ;    
}

float SGLM::getGazeLength() const { return glm::length(gaze) ; }   // must be after updateELU 


std::string SGLM::descELU() const 
{
    glm::mat4 escale = get_escale(); 
    std::stringstream ss ; 
    ss << "SGLM::descELU" << std::endl ; 
    ss << std::setw(15) << " sglm.EYE "  << Present( EYE )  << std::endl ; 
    ss << std::setw(15) << " sglm.LOOK " << Present( LOOK ) << std::endl ; 
    ss << std::setw(15) << " sglm.UP "   << Present( UP )   << std::endl ; 
    ss << std::setw(15) << " sglm.GAZE " << Present( LOOK-EYE ) << std::endl ; 
    ss << std::endl ; 
    ss << std::setw(15) << " sglm.EYE*escale "  << Present( EYE*escale )  << std::endl ; 
    ss << std::setw(15) << " sglm.LOOK*escale " << Present( LOOK*escale ) << std::endl ; 
    ss << std::setw(15) << " sglm.UP*escale "   << Present( UP*escale )   << std::endl ; 
    ss << std::setw(15) << " sglm.GAZE*escale " << Present( (LOOK-EYE)*escale ) << std::endl ; 
    ss << std::endl ; 
    ss << std::setw(15) << " sglm.eye "  << Present( eye )  << std::endl ; 
    ss << std::setw(15) << " sglm.look " << Present( look ) << std::endl ; 
    ss << std::setw(15) << " sglm.up "   << Present( up )   << std::endl ; 
    ss << std::setw(15) << " sglm.gaze " << Present( gaze ) << std::endl ; 
    ss << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
}


/**
SGLM::updateNear
------------------

As the basis needs gazelength, this must be done after updateELU
but isnt there still a problem of basis consistency between tmin_abs and set_near_abs ? 
For example extent scaling vs gazelength scaling ? 

**/
void SGLM::updateNear() 
{
    float tma = tmin_abs() ; 
    addlog("updateNear.tma", tma);
    set_near_abs(tma) ;  
}
std::string SGLM::descNear() const
{
    std::stringstream ss ; 
    ss << "SGLM::descNear" << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
}



/**
SGLM::updateEyeSpace
---------------------

Normalized eye space oriented via gaze and up directions.  

        +Y    -Z
     top_ax  forward_ax  
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
    first translates a world frame point to the eye point, 
    making eye point the origin then rotates to get into camera frame
    using the OpenGL eye space convention : -Z is forward, +X to right, +Y up

**/

void SGLM::updateEyeSpace()
{
    forward_ax = glm::normalize(gaze);
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

NB this uses the *camera2world* matrix obtained from SGLM::updateEyeSpace
with the same OpenGL eye frame convention to yield a scaled set of basis
vectors.  

The u,v,w,e, vectors are used from CSGOptiX::prepareRenderParam 
for setting up the raytrace render params. 

**/

void SGLM::updateEyeBasis()
{
    glm::vec4 rht( 1., 0., 0., 0.);  // +X
    glm::vec4 top( 0., 1., 0., 0.);  // +Y
    glm::vec4 gaz( 0., 0.,-1., 0.);  // -Z
    glm::vec4 ori( 0., 0., 0., 1.); 

    float aspect = Aspect() ; 
    float fsc = get_focal_basis() ; 
    float fscz = fsc/ZOOM  ; 
    float gazlen = getGazeLength() ;  // HMM: maybe get_nearfar_basis for consistency

    u = glm::vec3( camera2world * rht ) * fscz * aspect ;  
    v = glm::vec3( camera2world * top ) * fscz  ;  
    w = glm::vec3( camera2world * gaz ) * gazlen ;    
    e = glm::vec3( camera2world * ori );   
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
        case BAS_EXTENT:     basis = extent()            ; break ;  // only after setFrame 
        case BAS_GAZELENGTH: basis = getGazeLength()     ; break ;  // only available after updateELU (default)
        case BAS_NEARABS:    assert(0)                   ; break ;  // this mode only valud for get_focal_basis 
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
void SGLM::set_far_abs(  float far_abs_ ){  addlog("set_far_abs", far_abs_)   ; set_far(  far_abs_/get_nearfar_basis()  ) ; }

float SGLM::get_near_abs() const { return near*get_nearfar_basis() ; }
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

 
void SGLM::updateProjection()
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

    projection = glm::frustum( left, right, bottom, top, near_abs, far_abs );
    world2clip = projection * world2camera ;  //  ModelViewProjection :  no look rotation or trackballing   
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
    ss << std::setw(wid) << "sglm.world2clip\n" << Present(world2clip) << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
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
inline void SGLM::GetEVector(std::vector<T>& vec, const char* key, const char* fallback )  // static 
{
    const char* sval = getenv(key); 
    std::stringstream ss(sval ? sval : fallback); 
    std::string s ; 
    while(getline(ss, s, ',')) vec.push_back(ato_<T>(s.c_str()));   
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

template <typename T>
inline T SGLM::EValue( const char* key, const char* fallback )  // static 
{
    const char* sval = getenv(key); 
    std::string s(sval ? sval : fallback); 
    T value = ato_<T>(s.c_str());
    return value ;    
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



