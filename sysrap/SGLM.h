#pragma once
/**
SGLM : Header Only OpenGL viz projection maths 
=================================================

Normal inputs WH, CE, EYE, LOOK, UP 
are held in static variables with envvar defaults 
These can be changed with static methods before instanciating SGLM. 
NB it will usually be too late for setenv in code to influence SGLM 
as the static initialization would have happened already 
 

* https://learnopengl.com/Getting-started/Camera

TODO: WASD camera navigation, using a method intended to be called from the GLFW key callback 

TODO: provide a single header replacement for the boatload of classes used by okc/Composition

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

#include "SYSRAP_API_EXPORT.hh"


struct SYSRAP_API SGLM
{
    static SGLM* INSTANCE ; 

    static glm::ivec2 WH ; 
    static glm::vec4 CE ;    // static default, can be overridden by *ce* member 
    static glm::vec4 EYE ;    // model frame 
    static glm::vec4 LOOK ; 
    static glm::vec4 UP ; 
    static float ZOOM ; 
    static int   CAM ; 

    static constexpr const char* kWH = "WH" ; 
    static constexpr const char* kCE = "CE" ; 
    static constexpr const char* kEYE = "EYE" ; 
    static constexpr const char* kLOOK = "LOOK" ; 
    static constexpr const char* kUP = "UP" ; 
    static constexpr const char* kZOOM = "ZOOM" ; 
    static constexpr const char* kCAM = "CAM" ; 

    static void SetWH( int width, int height ); 
    static void SetCE( float x, float y, float z, float extent ); 
    static void SetEYE( float x, float y, float z ); 
    static void SetLOOK( float x, float y, float z ); 
    static void SetUP( float x, float y, float z ); 
    static void SetZOOM( float v ); 
    static void SetCAM( const char* cam ); 

    static std::string DescInput() ; 
    std::string descInput() const ; 

    static int Width() ; 
    static int Height() ; 
    static float Aspect(); 
    static const char* CAMLabel() ; 

    template <typename T> static T ato_( const char* a );
    template <typename T> static void GetEVector(std::vector<T>& vec, const char* key, const char* fallback );
    template <typename T> static std::string Present(std::vector<T>& vec);


    static std::string Present(const glm::ivec2& v, int wid=6 );
    static std::string Present(const float v, int wid=10, int prec=3);
    static std::string Present(const glm::vec3& v, int wid=10, int prec=3);
    static std::string Present(const glm::vec4& v, int wid=10, int prec=3);
    static std::string Present(const glm::mat4& m, int wid=10, int prec=3);

    template<typename T> static std::string Present_(const glm::tmat4x4<T>& m, int wid=10, int prec=3);

    static void GetEVec(glm::vec3& v, const char* key, const char* fallback );
    static void GetEVec(glm::vec4& v, const char* key, const char* fallback );

    template <typename T> static T EValue(const char* key, const char* fallback );
    static glm::ivec2 EVec2i(const char* key, const char* fallback); 
    static glm::vec3 EVec3(const char* key, const char* fallback); 
    static glm::vec4 EVec4(const char* key, const char* fallback, float missing=1.f ); 

    template<typename T> static glm::tmat4x4<T> DemoMatrix(T scale); 



    SGLM(); 

    glm::vec4 ce ; 
    int  cam ; 

    const qat4* m2w ; 
    const qat4* w2m ; 
    glm::mat4 model2world ; 
    glm::mat4 world2model ; 
    bool rtp_tangential ; 
    int updateModelMatrix_branch ; 

    void set_ce(  float x, float y, float z, float w ); 
    void set_m2w( const qat4* m2w_, const qat4* w2m_ ); 
    void set_rtp_tangential( bool rtp_tangential_ ); 

    void updateModelMatrix();  // depends on ce, unless valid m2w and w2m matrices provided


    float basis ; 
    float near ; 
    float far ; 

    void set_basis(float basis_ ) ; // when 0.f basis is taken from ce.w 
    void set_near( float near_ ); 
    void set_far( float far_ ); 

    void set_near_abs( float near_abs_ ); 
    void set_far_abs( float far_abs_ ); 

    void set_basis_to_gazelength() ;  // near/far are in units of basis  
    void set_basis_to_extent() ; 

    float get_basis() const ;  // when not set return default of ce.w  
    float get_near() const ;  
    float get_far() const ;  

    float get_near_abs() const ;  
    float get_far_abs() const ;  


    std::string descBasis() const ; 

    float focalscale ; 
    float orthoscale ; 

    glm::vec3 eye ;  // world frame  
    glm::vec3 look ; 
    glm::vec3 up ; 
    glm::vec3 gaze ; 

    void updateELU();   // depends on CE and EYE, LOOK, UP 
    std::string descELU() const ; 

    glm::vec3 forward_ax ; 
    glm::vec3 right_ax ; 
    glm::vec3 top_ax ; 
    glm::mat4 rot_ax ;  
    glm::mat4 world2camera ; 
    glm::mat4 camera2world ; 

    void updateEyeSpace(); 
    std::string descEyeSpace() const ; 
    static std::string DescEyeBasis( const glm::vec3& E, const glm::vec3& U, const glm::vec3& V, const glm::vec3& W ); 

    glm::vec3 u ; 
    glm::vec3 v ; 
    glm::vec3 w ; 
    glm::vec3 e ; 
    void updateEyeBasis(); 
    std::string descEyeBasis() const ; 


    glm::mat4 projection ; 
    glm::mat4 world2clip ; 

    float getFocalScaleDefault() const ; 
    float getFocalScale() const ; 
    void setFocalScale(float fsc) ; 
    void setOrthoScale(float osc) ; 
    void setFocalScaleToGazeLength() ; 
    float getGazeLength() const ; 

    void updateProjection(); 
    std::string descProjection() const ; 

    void update(); 
    std::string desc() const ; 
    void dump() const ; 
};

SGLM* SGLM::INSTANCE = nullptr ; 

glm::ivec2 SGLM::WH = EVec2i(kWH,"1920,1080") ; 
glm::vec4  SGLM::CE = EVec4(kCE,"0,0,0,100") ; 
glm::vec4  SGLM::EYE  = EVec4(kEYE, "-1,-1,0,1") ; 
glm::vec4  SGLM::LOOK = EVec4(kLOOK, "0,0,0,1") ; 
glm::vec4  SGLM::UP  =  EVec4(kUP,   "0,0,1,0") ; 
float      SGLM::ZOOM = EValue<float>(kZOOM, "1"); 
int        SGLM::CAM  = SCAM::EGet(kCAM, "perspective") ; 

void SGLM::SetWH( int width, int height ){ WH.x = width ; WH.y = height ; }
void SGLM::SetCE(  float x, float y, float z, float w){ CE.x = x ; CE.y = y ; CE.z = z ;  CE.w = w ; }
void SGLM::SetEYE( float x, float y, float z){ EYE.x = x  ; EYE.y = y  ; EYE.z = z  ;  EYE.w = 1.f ; }
void SGLM::SetLOOK(float x, float y, float z){ LOOK.x = x ; LOOK.y = y ; LOOK.z = z ;  LOOK.w = 1.f ; }
void SGLM::SetUP(  float x, float y, float z){ UP.x = x   ; UP.y = y   ; UP.z = z   ;  UP.w = 1.f ; }
void SGLM::SetZOOM( float v ){ ZOOM = v ; }
void SGLM::SetCAM( const char* cam ){ CAM = SCAM::Type(cam) ; }

int SGLM::Width(){  return WH.x ; }
int SGLM::Height(){ return WH.y ; }
float SGLM::Aspect() { return float(WH.x)/float(WH.y) ; } 
const char* SGLM::CAMLabel(){ return SCAM::Name(CAM) ; }

SGLM::SGLM() 
    :
    ce(CE),
    cam(CAM),
    m2w(nullptr),
    w2m(nullptr),
    model2world(1.f), 
    world2model(1.f),
    rtp_tangential(false),
    updateModelMatrix_branch(-1), 

    basis(0.f),
    near(0.1f),   // units of basis
    far(5.f),     // units of basis

    focalscale(0.f),
    orthoscale(1.f),

    forward_ax(0.f,0.f,0.f),
    right_ax(0.f,0.f,0.f),
    top_ax(0.f,0.f,0.f),
    rot_ax(1.f),
    projection(1.f),
    world2clip(1.f)
{
    update(); 
    INSTANCE = this ; 
}

void SGLM::update()  
{
    updateModelMatrix(); 
    updateELU(); 
    updateEyeSpace(); 
    updateEyeBasis(); 
    updateProjection(); 
}

std::string SGLM::desc() const 
{
    std::stringstream ss ; 
    ss << DescInput() << std::endl ; 
    ss << descInput() << std::endl ; 
    ss << descELU() << std::endl ; 
    ss << descEyeSpace() << std::endl ; 
    ss << descEyeBasis() << std::endl ; 
    ss << descProjection() << std::endl ; 
    ss << descBasis() << std::endl ; 
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
    ss << std::setw(15) << kCAM << " " << CAMLabel() << std::endl ; 
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
    ss << std::setw(25) << " sglm.ce "  << Present( ce )   << std::endl ; 
    ss << std::setw(25) << " sglm.cam " << cam << std::endl ; 
    ss << std::setw(25) << " SCAM::Name(sglm.cam) " << SCAM::Name(cam) << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
}
 


void SGLM::set_ce( float x, float y, float z, float w )
{
    ce.x = x ; 
    ce.y = y ; 
    ce.z = z ; 
    ce.w = w ; 
}

void SGLM::set_rtp_tangential(bool rtp_tangential_ )
{
    rtp_tangential = rtp_tangential_ ; 
}


void SGLM::set_m2w( const qat4* m2w_, const qat4* w2m_ )
{
    m2w = m2w_ ; 
    w2m = w2m_ ; 
}

/**
SGLM::updateModelMatrix
------------------------

Called by SGLM::update. The matrices are set based on center--extent OR directly from the set_m2w matrices 

**/

void SGLM::updateModelMatrix()
{
    updateModelMatrix_branch = 0 ; 

    bool m2w_valid = m2w && m2w->is_identity() == false ;
    bool w2m_valid = w2m && w2m->is_identity() == false ;

    if( m2w_valid && w2m_valid )
    {
        updateModelMatrix_branch = 1 ; 
        model2world = glm::make_mat4x4<float>(m2w->cdata());
        world2model = glm::make_mat4x4<float>(w2m->cdata());
    }
    else if( rtp_tangential )
    {
        updateModelMatrix_branch = 2 ; 
        SCenterExtentFrame<double> cef( ce.x, ce.y, ce.z, ce.w, rtp_tangential );
        model2world = cef.model2world ;
        world2model = cef.world2model ;
    }
    else
    {
        updateModelMatrix_branch = 3 ; 
        glm::vec3 tr(ce.x, ce.y, ce.z) ;  
        glm::vec3 sc(ce.w, ce.w, ce.w) ; 
        glm::vec3 isc(1.f/ce.w, 1.f/ce.w, 1.f/ce.w) ; 

        model2world = glm::scale(glm::translate(glm::mat4(1.0), tr), sc);
        world2model = glm::translate( glm::scale(glm::mat4(1.0), isc), -tr); 
    }
}

void SGLM::set_basis(float basis_){ basis = basis_ ; }
void SGLM::set_basis_to_gazelength() { set_basis( getGazeLength()) ; }
void SGLM::set_basis_to_extent() {     set_basis( ce.w ) ; }

float SGLM::get_basis() const { return basis == 0.f ? ce.w : basis ; }


void SGLM::set_near( float near_ ){ near = near_ ; }
void SGLM::set_far(  float far_ ){  far = far_ ; }

float SGLM::get_near() const  { return near ; }  
float SGLM::get_far()  const  { return far  ; }  

void SGLM::set_near_abs( float near_abs_ ){ near = near_abs_/get_basis()  ; }
void SGLM::set_far_abs(  float far_abs_ ){  far = far_abs_/get_basis()    ; }

float SGLM::get_near_abs() const { return near*get_basis() ; }
float SGLM::get_far_abs() const { return   far*get_basis() ; }


std::string SGLM::descBasis() const 
{
    int wid = 25 ; 
    std::stringstream ss ; 
    ss << "SGLM::descBasis" << std::endl ; 
    ss << std::setw(wid) << " sglm.ce.w  "     << Present( ce.w )  << std::endl ; 
    ss << std::setw(wid) << " sglm.getGazeLength  " << Present( getGazeLength() ) << std::endl ; 
    ss << std::setw(wid) << " sglm.basis "     << Present( basis ) << std::endl ; 
    ss << std::setw(wid) << " sglm.get_basis " << Present( get_basis() ) << std::endl ; 
    ss << std::setw(wid) << " sglm.near  "     << Present( near )  << " (units of basis) " << std::endl ; 
    ss << std::setw(wid) << " sglm.far   "     << Present( far )   << " (units of basis) " << std::endl ; 
    ss << std::setw(wid) << " sglm.get_near_abs  " << Present( get_near_abs() ) << std::endl ; 
    ss << std::setw(wid) << " sglm.get_far_abs  " << Present( get_far_abs() ) << std::endl ; 
    ss << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
}

void SGLM::updateELU() // eye, look, up, gaze into world frame 
{
    eye  = glm::vec3( model2world * EYE ) ; 
    look = glm::vec3( model2world * LOOK ) ; 
    up   = glm::vec3( model2world * UP ) ; 
    gaze = glm::vec3( model2world * (LOOK - EYE) ) ;    
}

std::string SGLM::descELU() const 
{
    std::stringstream ss ; 
    ss << "SGLM::descELU" << std::endl ; 
    ss << " sglm.model2world \n" << Present( model2world ) << std::endl ; 
    ss << " sglm.world2model \n" << Present( world2model ) << std::endl ; 
    ss << " sglm.updateModelMatrix_branch " << updateModelMatrix_branch << std::endl ; 
    ss << std::endl ; 
    ss << std::setw(15) << " sglm.EYE "  << Present( EYE )  << std::endl ; 
    ss << std::setw(15) << " sglm.LOOK " << Present( LOOK ) << std::endl ; 
    ss << std::setw(15) << " sglm.UP "   << Present( UP )   << std::endl ; 
    ss << std::setw(15) << " sglm.GAZE " << Present( LOOK-EYE ) << std::endl ; 
    ss << std::endl ; 
    ss << std::setw(15) << " sglm.eye "  << Present( eye )  << std::endl ; 
    ss << std::setw(15) << " sglm.look " << Present( look ) << std::endl ; 
    ss << std::setw(15) << " sglm.up "   << Present( up )   << std::endl ; 
    ss << std::setw(15) << " sglm.gaze " << Present( gaze ) << std::endl ; 
    ss << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
}


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


float SGLM::getFocalScaleDefault() const { return (cam == CAM_ORTHOGRAPHIC ? orthoscale : get_near_abs() )/ZOOM ; } 
float SGLM::getFocalScale() const { return focalscale > 0 ? focalscale : getFocalScaleDefault() ; } 
void  SGLM::setFocalScale(float focalscale_) { focalscale = focalscale_ ; }
void  SGLM::setOrthoScale(float orthoscale_) { orthoscale = orthoscale_ ; }

void SGLM::setFocalScaleToGazeLength(){ setFocalScale( getGazeLength() ) ; }
float SGLM::getGazeLength() const { return glm::length(gaze) ; } 

void SGLM::updateProjection()
{
    float fsc = getFocalScale() ; 
    float aspect = Aspect(); 

    float left   = -aspect*fsc ;
    float right  =  aspect*fsc ;
    float bottom = -fsc ;
    float top    =  fsc ;

    float near_abs   = get_near_abs() ; 
    float far_abs    = get_far_abs()  ; 

    projection = glm::frustum( left, right, bottom, top, near_abs, far_abs );

    world2clip = projection * world2camera ;  //  ModelViewProjection :  no look rotation or trackballing   
}



std::string SGLM::descProjection() const 
{
    float fsc = getFocalScale() ; 
    float aspect = Aspect(); 
    float left   = -aspect*fsc ;
    float right  =  aspect*fsc ;
    float bottom = -fsc ;
    float top    =  fsc ;
    float near_abs   = get_near_abs() ; 
    float far_abs    = get_far_abs()  ; 


    int wid = 25 ; 
    std::stringstream ss ;
    ss << "SGLM::descProjection" << std::endl ; 
    ss << std::setw(wid) << "Aspect" << Present(aspect) << std::endl ;  
    ss << std::setw(wid) << "getFocalScale" << Present(fsc) << std::endl ;  
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


/**
SGLM::updateEyeBasis
---------------------

HMM: to match Composition have to use focalplane_scale of the gazelength  
but have been using scale of near/ZOOM 

**/

void SGLM::updateEyeBasis()
{
    glm::vec4 rht( 1., 0., 0., 0.); 
    glm::vec4 top( 0., 1., 0., 0.); 
    glm::vec4 gaz( 0., 0.,-1., 0.);   // towards -Z
    glm::vec4 ori( 0., 0., 0., 1.); 

    float aspect = Aspect() ; 
    float gazlen = getGazeLength() ; 
    float fsc = getFocalScale() ; 

    u = glm::vec3( camera2world * rht ) * fsc * aspect ;  
    v = glm::vec3( camera2world * top ) * fsc  ;  
    w = glm::vec3( camera2world * gaz ) * gazlen ;    
    e = glm::vec3( camera2world * ori );   
}

std::string SGLM::descEyeBasis() const 
{
    std::stringstream ss ;
    ss << "SGLM::descEyeBasis : camera frame basis vectors transformed into world and focal plane scaled " << std::endl ; 

    int wid = 25 ; 
    float aspect = Aspect() ; 
    float fscd = getFocalScaleDefault() ;
    float fsc = getFocalScale() ;
    float gazlen = getGazeLength() ; 

    ss << std::setw(wid) << "aspect" << Present(aspect) << std::endl ;
    ss << std::setw(wid) << "near " << Present(near) << std::endl ;
    ss << std::setw(wid) << "ZOOM " << Present(ZOOM) << std::endl ;
    ss << std::setw(wid) << "getFocalScaleDefault " << Present(fscd) << std::endl ;
    ss << std::setw(wid) << "getFocalScale:fsc" << Present(fsc) << std::endl ;
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



