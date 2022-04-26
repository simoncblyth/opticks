#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/string_cast.hpp>


#include "SYSRAP_API_EXPORT.hh"

/**
SGLM

* https://learnopengl.com/Getting-started/Camera

TODO: WASD camera navigation, using a method intended to be called from the GLFW key callback 

**/

struct SYSRAP_API SGLM
{
    static SGLM* INSTANCE ; 

    template <typename T>
    static T ato_( const char* a );

    template <typename T>
    static void GetEVector(std::vector<T>& vec, const char* key, const char* fallback );

    template <typename T>
    static std::string Present(std::vector<T>& vec);

    static std::string Present(const glm::vec3& v, int wid=10, int prec=3);
    static std::string Present(const glm::vec4& v, int wid=10, int prec=3);
    static std::string Present(const glm::mat4& m, int wid=10, int prec=3);

    static void GetEVec(glm::vec3& v, const char* key, const char* fallback );
    static void GetEVec(glm::vec4& v, const char* key, const char* fallback );

    static glm::vec3 EVec3(const char* key, const char* fallback); 
    static glm::vec4 EVec4(const char* key, const char* fallback); 

    static glm::mat4 GetMVP(int width, int height, bool verbose); 



    SGLM(); 

    int width ;        // inputs 
    int height ; 
    glm::vec4 ce ;
    glm::vec4 eye_m ;   
    glm::vec4 look_m ; 
    glm::vec4 up_m ; 

    std::string descInput() const ; 

    void setCenter(float x, float y, float z);
    void setExtent(float extent); 


    glm::vec3 tr ;           // derived from ce 
    glm::vec3 sc ; 
    glm::vec3 isc ; 
    glm::mat4 model2world ; 
    glm::mat4 world2model ; 

    glm::vec3 eye ; 
    glm::vec3 look ; 
    glm::vec3 up ; 
    glm::vec3 gaze ; 


    void updateModelMatrix(); 
    void updateELU(); 
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
    float aspect ; 
    float zoom ;
    float scale ; 
    float near ; 
    float far ; 
    float left ; 
    float right ; 
    float bottom ; 
    float top ; 
    glm::mat4 projection ; 
    glm::mat4 world2clip ; 

    void updateProjection(); 
    std::string descProjection() const ; 

    void update(); 
    std::string desc() const ; 
    void dump() const ; 
};

SGLM* SGLM::INSTANCE = nullptr ; 


SGLM::SGLM() 
    :
    width(1024),
    height(768),

    ce(0.f, 0.f, 0.f, 100.f),
    eye_m( -1.f, -1.f, 0.f, 1.f),
    look_m( 0.f,  0.f, 0.f, 1.f),
    up_m(   0.f,  0.f, 1.f, 0.f),

    tr(0.f, 0.f, 0.f),
    sc(1.f, 1.f, 1.f),
    isc(1.f, 1.f, 1.f),
    model2world(1.f), 
    world2model(1.f),

    forward_ax(0.f,0.f,0.f),
    right_ax(0.f,0.f,0.f),
    top_ax(0.f,0.f,0.f),
    rot_ax(1.f),

    parallel(false),
    orthoscale(1.f),
    aspect(float(width)/float(height)),
    zoom(1.f), 
    scale(1.f), 
    near(0.f), 
    far(0.f),
    left(0.f), 
    right(0.f), 
    bottom(0.f),
    top(0.f),
    projection(1.f),
    world2clip(1.f)
{
    setCenter(ce.x, ce.y, ce.z); 
    setExtent(ce.w); 

    update(); 
    INSTANCE = this ; 
}

void SGLM::update()  
{
    // *dirty*  toggle a bit pointless as changing almost any member would require update  
    updateModelMatrix(); 
    updateELU(); 
    updateEyeSpace(); 
    updateProjection(); 
}

std::string SGLM::desc() const 
{
    std::stringstream ss ; 
    ss << descInput() << std::endl ; 
    ss << descELU() << std::endl ; 
    ss << descEyeSpace() << std::endl ; 
    ss << descProjection() << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
}

void SGLM::dump() const 
{
    std::cout << desc() << std::endl ; 
}

void SGLM::setCenter(float x, float y, float z)
{
    ce.x = x ; 
    ce.y = y ; 
    ce.z = z ; 

    tr.x = ce.x ; 
    tr.y = ce.y ; 
    tr.z = ce.z ; 
}

void SGLM::setExtent(float extent)  
{
    ce.w = extent ; 

    sc.x = ce.w ; 
    sc.y = ce.w ; 
    sc.z = ce.w ; 

    isc.x = 1.f/ce.w ; 
    isc.y = 1.f/ce.w ; 
    isc.z = 1.f/ce.w ; 
}

std::string SGLM::descInput() const 
{
    std::stringstream ss ; 
    ss << "SGLM::descInput" << std::endl ; 
    ss << std::setw(15) << " ce "      << Present( ce )     << std::endl ; 
    ss << std::setw(15) << " eye_m "   << Present( eye_m )  << std::endl ; 
    ss << std::setw(15) << " look_m "  << Present( look_m ) << std::endl ; 
    ss << std::setw(15) << " up_m "    << Present( up_m )   << std::endl ; 
    ss << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
}

void SGLM::updateModelMatrix()
{
    model2world = glm::scale(glm::translate(glm::mat4(1.0), tr), sc);
    world2model = glm::translate( glm::scale(glm::mat4(1.0), isc), -tr); 
}

void SGLM::updateELU()
{
    const glm::mat4& m2w = model2world ; 
    eye = glm::vec3( m2w * eye_m ) ; 
    look = glm::vec3( m2w * look_m ) ; 
    up = glm::vec3( m2w * up_m ) ; 
    gaze = glm::vec3( m2w * (look_m - eye_m) ) ;    
}

std::string SGLM::descELU() const 
{
    std::stringstream ss ; 
    ss << "SGLM::descELU" << std::endl ; 
    ss << " model2world \n" << Present( model2world ) << std::endl ; 
    ss << " world2model \n" << Present( world2model ) << std::endl ; 
    ss << std::setw(15) << " eye "  << Present( eye )  << std::endl ; 
    ss << std::setw(15) << " look " << Present( look ) << std::endl ; 
    ss << std::setw(15) << " up "   << Present( up )   << std::endl ; 
    ss << std::setw(15) << " gaze " << Present( gaze ) << std::endl ; 
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

    glm::mat4 ti(glm::translate(glm::vec3(eye)));
    glm::mat4 t(glm::translate(glm::vec3(-eye)));  // eye to origin 

    world2camera = glm::transpose(rot_ax) * t  ;
    camera2world = ti * rot_ax ;
}

std::string SGLM::descEyeSpace() const 
{
    std::stringstream ss ; 
    ss << "SGLM::descEyeSpace" << std::endl ; 
    ss << std::setw(15) << "forward_ax" << Present(forward_ax) << std::endl ;  
    ss << std::setw(15) << "right_ax" << Present(right_ax) << std::endl ;  
    ss << std::setw(15) << "top_ax" << Present(top_ax) << std::endl ;  
    ss << std::endl ; 

    ss << " world2camera \n" << Present( world2camera ) << std::endl ; 
    ss << " camera2world \n" << Present( camera2world ) << std::endl ; 
    ss << std::endl ; 

    std::string s = ss.str(); 
    return s ; 
}


void SGLM::updateProjection()
{
    aspect = float(width)/float(height) ;

    // Camera::aim
    float basis = ce.w ;
    near = basis/10.f ;
    far = basis*5.f ;

    // Camera::getFrustum
    scale = parallel ? orthoscale : near   ;

    left = -aspect*scale/zoom ;
    right = aspect*scale/zoom ;

    bottom = -scale/zoom ;
    top = scale/zoom ;

    projection = glm::frustum( left, right, bottom, top, near, far );


    const glm::mat4& world2eye = world2camera ; // no look rotation or trackballing  
    world2clip = projection * world2eye ;    //  ModelViewProjection 
}

std::string SGLM::descProjection() const 
{
    std::stringstream ss ;
    ss << "SGLM::descProjection" << std::endl ; 
    ss << std::setw(15) << "aspect" << std::setw(10) << std::fixed << std::setprecision(4) << aspect << std::endl ;  
    ss << std::setw(15) << "scale" << std::setw(10) << std::fixed << std::setprecision(4) << scale << std::endl ;  
    ss << std::setw(15) << "zoom" << std::setw(10) << std::fixed << std::setprecision(4) << zoom << std::endl ;  
    ss << std::setw(15) << "near" << std::setw(10) << std::fixed << std::setprecision(4) << near << std::endl ;  
    ss << std::setw(15) << "far" << std::setw(10) << std::fixed << std::setprecision(4) << far << std::endl ;  
    ss << std::setw(15) << "left" << std::setw(10) << std::fixed << std::setprecision(4) << left << std::endl ;  
    ss << std::setw(15) << "right" << std::setw(10) << std::fixed << std::setprecision(4) << right << std::endl ;  
    ss << std::setw(15) << "bottom" << std::setw(10) << std::fixed << std::setprecision(4) << bottom << std::endl ;  
    ss << std::setw(15) << "top" << std::setw(10) << std::fixed << std::setprecision(4) << top << std::endl ;  
    ss << std::setw(15) << "projection\n" << Present(projection) << std::endl ; 
    ss << std::setw(15) << "world2clip\n" << Present(world2clip) << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
}



/**
SGLM::GetMVP
-------------

Monolithic approach, should be duplicated by the above more maintainable approach. 

**/
inline glm::mat4 SGLM::GetMVP(int width, int height, bool verbose)  // static
{
    // Composition::setCenterExtent 
    glm::vec4 ce(0.f, 0.f, 0.f, 100.f );  // center extent of the "model"

    glm::vec3 tr(ce.x, ce.y, ce.z);
    glm::vec3 sc(ce.w);
    glm::vec3 isc(1.f/ce.w);

    glm::mat4 model2world = glm::scale(glm::translate(glm::mat4(1.0), tr), sc);
    glm::mat4 world2model = glm::translate( glm::scale(glm::mat4(1.0), isc), -tr); 

    // View::getTransforms
    glm::vec4 eye_m( -1.f,-1.f,0.f,1.f);   //  viewpoint in unit model frame 
    glm::vec4 look_m( 0.f, 0.f,0.f,1.f); 
    glm::vec4 up_m(   0.f, 0.f,1.f,1.f); 
    glm::vec4 gze_m( look_m - eye_m ) ; 

    const glm::mat4& m2w = model2world ; 
    glm::vec3 eye = glm::vec3( m2w * eye_m ) ; 
    glm::vec3 look = glm::vec3( m2w * look_m ) ; 
    glm::vec3 up = glm::vec3( m2w * up_m ) ; 
    glm::vec3 gaze = glm::vec3( m2w * gze_m ) ;    

    if(verbose)
    {   
       std::cout << " model2world \n" << Present( model2world ) << std::endl ; 
       std::cout << " world2model \n" << Present( world2model ) << std::endl ; 
       std::cout << " eye \n"         << Present( eye )         << std::endl ; 
       std::cout << " look \n"        << Present( look )        << std::endl ; 
       std::cout << " up \n"          << Present( up )          << std::endl ; 
       std::cout << " gaze \n"        << Present( gaze )        << std::endl ; 
    }   


    glm::vec3 forward_ax = glm::normalize(gaze);
    glm::vec3 right_ax   = glm::normalize(glm::cross(forward_ax,up)); 
    glm::vec3 top_ax     = glm::normalize(glm::cross(right_ax,forward_ax));

    // OpenGL eye space convention : -Z is forward, +X to right, +Y up
    glm::mat4 rot(1.0) ; 
    rot[0] = glm::vec4( right_ax, 0.f );  
    rot[1] = glm::vec4( top_ax  , 0.f );  
    rot[2] = glm::vec4( -forward_ax, 0.f );  


    glm::mat4 ti(glm::translate(glm::vec3(eye)));
    glm::mat4 t(glm::translate(glm::vec3(-eye)));  // eye to origin 

    glm::mat4 world2camera = glm::transpose(rot) * t  ;
    glm::mat4 camera2world = ti * rot ;



    float gazelength = glm::length(gaze);
    glm::mat4 eye2look = glm::translate( glm::mat4(1.), glm::vec3(0,0,gazelength));
    glm::mat4 look2eye = glm::translate( glm::mat4(1.), glm::vec3(0,0,-gazelength));


    //glm::vec4 gaze = glm::vec4( gze, 0.f );

    // Composition::update
    glm::mat4 world2eye = world2camera ; // no look rotation or trackballing  

    float aspect = float(width)/float(height) ;

    // Camera::aim
    float basis = 100.f ;
    float near = basis/10.f ;
    float far = basis*5.f ;
    float zoom = 1.0f ;

    // Camera::getFrustum
    bool parallel = false ;
    float orthoscale = 1.0f ;
    float scale = parallel ? orthoscale : near   ;

    float left = -aspect*scale/zoom ;
    float right = aspect*scale/zoom ;

    float top = scale/zoom ;
    float bottom = -scale/zoom ;

    glm::mat4 projection = glm::frustum( left, right, bottom, top, near, far );
    glm::mat4 world2clip = projection * world2eye ;    //  ModelViewProjection 

    if(verbose)
    {
       std::cout << " rot  \n"         << Present( rot ) << std::endl ;
       std::cout << " eye2look  \n"    << Present( eye2look ) << std::endl ;
       std::cout << " look2eye  \n"    << Present( look2eye ) << std::endl ;
       std::cout << " world2camera \n" << Present( world2camera ) << std::endl ;
       std::cout << " camera2world \n" << Present( camera2world ) << std::endl ;
       std::cout << " projection \n"   << Present( projection ) << std::endl ;
       std::cout << " world2clip \n"   << Present( world2clip ) << std::endl ;
    }

    return world2clip ;
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

inline glm::vec4 SGLM::EVec4(const char* key, const char* fallback) // static
{
    std::vector<float> vec ; 
    SGLM::GetEVector<float>(vec, key, fallback); 
    glm::vec4 v ; 
    for(int i=0 ; i < 4 ; i++) v[i] = i < int(vec.size()) ? vec[i] : 0.f  ; 
    return v ; 
}

inline glm::vec3 SGLM::EVec3(const char* key, const char* fallback) // static
{
    std::vector<float> vec ; 
    SGLM::GetEVector<float>(vec, key, fallback); 
    glm::vec3 v ; 
    for(int i=0 ; i < 3 ; i++) v[i] = i < int(vec.size()) ? vec[i] : 0.f  ; 
    return v ; 
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





