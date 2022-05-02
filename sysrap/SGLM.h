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

    static constexpr const char* kWH = "WH" ; 
    static constexpr const char* kCE = "CE" ; 
    static constexpr const char* kEYE = "EYE" ; 
    static constexpr const char* kLOOK = "LOOK" ; 
    static constexpr const char* kUP = "UP" ; 
    static constexpr const char* kZOOM = "ZOOM" ; 

    static void SetWH( int width, int height ); 
    static void SetCE( float x, float y, float z, float extent ); 
    static void SetEYE( float x, float y, float z ); 
    static void SetLOOK( float x, float y, float z ); 
    static void SetUP( float x, float y, float z ); 
    static void SetZOOM( float v ); 

    static std::string DescInput() ; 

    static int Width() ; 
    static int Height() ; 
    static float Aspect(); 

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
    const qat4* m2w ; 
    const qat4* w2m ; 
    glm::mat4 model2world ; 
    glm::mat4 world2model ; 
    bool rtp_tangential ; 

    void set_ce(  float x, float y, float z, float w ); 
    void set_m2w( const qat4* m2w_, const qat4* w2m_ ); 
    void set_rtp_tangential( bool rtp_tangential_ ); 

    void updateModelMatrix();  // depends on ce, unless valid m2w and w2m matrices provided


    float basis ; 
    float near ; 
    float far ; 
    void set_basis(float basis_=0.f) ; // when 0.f basis is taken from ce.w 

    float scale ; 

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

    bool parallel ;  
    float orthoscale ; 

    glm::vec3 u ; 
    glm::vec3 v ; 
    glm::vec3 w ; 
    glm::vec3 e ; 
    void updateEyeBasis(); 
    std::string descEyeBasis() const ; 


    glm::mat4 projection ; 
    glm::mat4 world2clip ; 

    float getScaleDefault() const ; 
    float getScale() const ; 
    void setScale(float scale) ; 

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

void SGLM::SetWH( int width, int height ){ WH.x = width ; WH.y = height ; }
void SGLM::SetCE(  float x, float y, float z, float w){ CE.x = x ; CE.y = y ; CE.z = z ;  CE.w = w ; }
void SGLM::SetEYE( float x, float y, float z){ EYE.x = x  ; EYE.y = y  ; EYE.z = z  ;  EYE.w = 1.f ; }
void SGLM::SetLOOK(float x, float y, float z){ LOOK.x = x ; LOOK.y = y ; LOOK.z = z ;  LOOK.w = 1.f ; }
void SGLM::SetUP(  float x, float y, float z){ UP.x = x   ; UP.y = y   ; UP.z = z   ;  UP.w = 1.f ; }
void SGLM::SetZOOM( float v ){ ZOOM = v ; }

int SGLM::Width(){  return WH.x ; }
int SGLM::Height(){ return WH.y ; }
float SGLM::Aspect() { return float(WH.x)/float(WH.y) ; } 



SGLM::SGLM() 
    :
    ce(CE),
    m2w(nullptr),
    w2m(nullptr),
    model2world(1.f), 
    world2model(1.f),
    rtp_tangential(false),

    basis(ce.w),
    near(basis/10.f), 
    far(basis*5.f),

    scale(0.f),

    forward_ax(0.f,0.f,0.f),
    right_ax(0.f,0.f,0.f),
    top_ax(0.f,0.f,0.f),
    rot_ax(1.f),

    parallel(false),
    orthoscale(1.f),

    projection(1.f),
    world2clip(1.f)
{
    update(); 
    INSTANCE = this ; 
}

void SGLM::update()  
{
    updateModelMatrix(); 
    set_basis(); 

    updateELU(); 
    updateEyeSpace(); 
    updateEyeBasis(); 
    updateProjection(); 
}

std::string SGLM::desc() const 
{
    std::stringstream ss ; 
    ss << DescInput() << std::endl ; 
    ss << descELU() << std::endl ; 
    ss << descEyeSpace() << std::endl ; 
    ss << descEyeBasis() << std::endl ; 
    ss << descProjection() << std::endl ; 
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
    if( m2w && w2m )
    {
        model2world = glm::make_mat4x4<float>(m2w->cdata());
        world2model = glm::make_mat4x4<float>(w2m->cdata());
    }
    else if( rtp_tangential )
    {
        SCenterExtentFrame<double> cef( ce.x, ce.y, ce.z, ce.w, rtp_tangential );
        model2world = cef.model2world ;
        world2model = cef.world2model ;
    }
    else
    {
        glm::vec3 tr(ce.x, ce.y, ce.z) ;  
        glm::vec3 sc(ce.w, ce.w, ce.w) ; 
        glm::vec3 isc(1.f/ce.w, 1.f/ce.w, 1.f/ce.w) ; 

        model2world = glm::scale(glm::translate(glm::mat4(1.0), tr), sc);
        world2model = glm::translate( glm::scale(glm::mat4(1.0), isc), -tr); 
    }
}

void SGLM::set_basis(float basis_)
{
    basis = basis_ == 0 ? ce.w : basis_ ;
    near = basis/10.f ;
    far = basis*5.f ;
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

float SGLM::getScaleDefault() const { return (parallel ? orthoscale : near )/ZOOM ; } 
float SGLM::getScale() const { return scale > 0 ? scale : getScaleDefault() ; } 
void  SGLM::setScale(float scale_) { scale = scale_ ; }


void SGLM::updateProjection()
{
    float sc = getScale() ; 

    float aspect = Aspect(); 
    float left   = -aspect*sc ;
    float right  =  aspect*sc ;
    float bottom = -sc ;
    float top    =  sc ;

    projection = glm::frustum( left, right, bottom, top, near, far );

    const glm::mat4& world2eye = world2camera ; // no look rotation or trackballing  
    world2clip = projection * world2eye ;       //  ModelViewProjection 
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
    float sc = getScale() ; 
    float gazlen = glm::length(gaze) ; 

    float focalplane_scale = sc ; 
    //float focalplane_scale = gazlen ; 

    u = glm::vec3( camera2world * rht ) * focalplane_scale * aspect ;  
    v = glm::vec3( camera2world * top ) * focalplane_scale  ;  
    w = glm::vec3( camera2world * gaz ) * gazlen ;    
    e = glm::vec3( camera2world * ori );   
}

std::string SGLM::descEyeBasis() const 
{
    std::stringstream ss ;
    ss << "SGLM::descEyeBasis : camera frame basis vectors transformed into world and focal plane scaled " << std::endl ; 
    int wid = 15 ; 

    float aspect = Aspect() ; 
    float sc = getScale() ;
    float gazlen = glm::length(gaze) ; 

    ss << std::setw(wid) << "aspect" << Present(aspect) << std::endl ;
    ss << std::setw(wid) << "near " << Present(near) << std::endl ;
    ss << std::setw(wid) << "ZOOM " << Present(ZOOM) << std::endl ;
    ss << std::setw(wid) << "scale " << Present(sc) << " (parallel ? orthoscale : near )/ZOOM " << std::endl ;
    ss << std::setw(wid) << "gazlen " << Present(gazlen) << std::endl ;
    ss << std::setw(wid) << "sglm.u " << Present(u) << " glm::vec3( camera2world * rht ) * sc * aspect " << std::endl ; 
    ss << std::setw(wid) << "sglm.v " << Present(v) << " glm::vec3( camera2world * top ) * sc  " << std::endl ; 
    ss << std::setw(wid) << "sglm.w " << Present(w) << " glm::vec3( camera2world * gaz ) * gazlen  " << std::endl ; 
    ss << std::setw(wid) << "sglm.e " << Present(e) << " glm::vec3( camera2world * ori ) " << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
}

std::string SGLM::descProjection() const 
{
    int wid = 15 ; 
    std::stringstream ss ;
    ss << "SGLM::descProjection" << std::endl ; 
    ss << std::setw(wid) << "Aspect" << std::setw(10) << std::fixed << std::setprecision(4) << Aspect() << std::endl ;  
    ss << std::setw(wid) << "scale" << std::setw(10) << std::fixed << std::setprecision(4) << getScale() << std::endl ;  
    ss << std::setw(wid) << "ZOOM" << std::setw(10) << std::fixed << std::setprecision(4) << ZOOM << std::endl ;  
    ss << std::setw(wid) << "near" << std::setw(10) << std::fixed << std::setprecision(4) << near << std::endl ;  
    ss << std::setw(wid) << "far" << std::setw(10) << std::fixed << std::setprecision(4) << far << std::endl ;  
    ss << std::setw(wid) << "sglm.projection\n" << Present(projection) << std::endl ; 
    ss << std::setw(wid) << "sglm.world2clip\n" << Present(world2clip) << std::endl ; 
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



