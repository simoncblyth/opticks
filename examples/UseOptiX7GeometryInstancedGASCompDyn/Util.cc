#include <sstream>
#include <string>
#include <cstring>
#include <cstdlib>
#include "Util.h"
#include <glm/gtx/transform.hpp>

const char* Util::PTXPath( const char* install_prefix, const char* cmake_target, const char* cu_stem, const char* cu_ext ) // static
{
    std::stringstream ss ; 
    ss << install_prefix
       << "/ptx/"
       << cmake_target
       << "_generated_"
       << cu_stem
       << cu_ext
       << ".ptx" 
       ;
    std::string path = ss.str();
    return strdup(path.c_str()); 
}

const char* Util::PPMPath( const char* install_prefix, const char* stem, const char* ext ) // static 
{
    std::stringstream ss ; 
    ss << install_prefix
       << "/ppm/"
       << stem
       << ext
       ;
    std::string path = ss.str();
    return strdup(path.c_str()); 
}

const char* Util::NPYPath( const char* install_prefix, const char* stem, const char* ext ) // static 
{
    std::stringstream ss ; 
    ss << install_prefix
       << "/npy/"
       << stem
       << ext
       ;
    std::string path = ss.str();
    return strdup(path.c_str()); 
}


// Composition::getEyeUVW and examples/UseGeometryShader:getMVP
void Util::GetEyeUVW(const glm::vec4& ce, const unsigned width, const unsigned height, glm::vec3& eye, glm::vec3& U, glm::vec3& V, glm::vec3& W ) // static
{
    glm::vec3 tr(ce.x, ce.y, ce.z);  // ce is center-extent of model
    glm::vec3 sc(ce.w);
    glm::vec3 isc(1.f/ce.w);
    // model frame unit coordinates from/to world 
    glm::mat4 model2world = glm::scale(glm::translate(glm::mat4(1.0), tr), sc);
    //glm::mat4 world2model = glm::translate( glm::scale(glm::mat4(1.0), isc), -tr);

   // View::getTransforms
    glm::vec4 eye_m( -1.f,-1.f,1.f,1.f);  //  viewpoint in unit model frame 
    glm::vec4 look_m( 0.f, 0.f,0.f,1.f); 
    glm::vec4 up_m(   0.f, 0.f,1.f,1.f); 
    glm::vec4 gze_m( look_m - eye_m ) ; 

    const glm::mat4& m2w = model2world ; 
    glm::vec3 eye_ = glm::vec3( m2w * eye_m ) ; 
    //glm::vec3 look = glm::vec3( m2w * look_m ) ; 
    glm::vec3 up = glm::vec3( m2w * up_m ) ; 
    glm::vec3 gaze = glm::vec3( m2w * gze_m ) ;    

    glm::vec3 forward_ax = glm::normalize(gaze);
    glm::vec3 right_ax   = glm::normalize(glm::cross(forward_ax,up)); 
    glm::vec3 top_ax     = glm::normalize(glm::cross(right_ax,forward_ax));

    float aspect = float(width)/float(height) ;
    float tanYfov = 1.f ;  // reciprocal of camera zoom
    float gazelength = glm::length( gaze ) ;
    float v_half_height = gazelength * tanYfov ;
    float u_half_width  = v_half_height * aspect ;

    U = right_ax * u_half_width ;
    V = top_ax * v_half_height ;
    W = forward_ax * gazelength ; 
    eye = eye_ ; 
}


template <typename T>
T Util::ato_( const char* a )   // static 
{
    std::string s(a);
    std::istringstream iss(s);
    T v ; 
    iss >> v ; 
    return v ; 
}


template <typename T>
T Util::GetEValue(const char* key, T fallback) // static 
{
    const char* sval = getenv(key); 
    T val = sval ? ato_<T>(sval) : fallback ;
    return val ;  
}

template float    Util::GetEValue<float>(const char* key, float fallback); 
template int      Util::GetEValue<int>(const char* key,   int  fallback); 
template unsigned Util::GetEValue<unsigned>(const char* key,   unsigned  fallback); 

