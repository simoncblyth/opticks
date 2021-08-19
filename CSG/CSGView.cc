
#include <iostream>
#include <sstream>
#include <iomanip>

#include <glm/gtx/transform.hpp>
#include "NP.hh"
#include "CSGView.h"

void CSGView::update(const glm::vec4& em, const glm::vec4& ce, const unsigned width, const unsigned height)
{
    eye_model = em ; 
    center_extent = ce ; 

    glm::vec3 tr(ce.x, ce.y, ce.z);  // ce is center-extent of model
    glm::vec3 sc(ce.w);
    glm::vec3 isc(1.f/ce.w);
    // model frame unit coordinates from/to world 
    glm::mat4 model2world = glm::scale(glm::translate(glm::mat4(1.0), tr), sc);
    //glm::mat4 world2model = glm::translate( glm::scale(glm::mat4(1.0), isc), -tr);

   // CSGView::getTransforms
    glm::vec4 eye_m( em.x,em.y,em.z,1.f);  //  viewpoint in unit model frame 
    glm::vec4 look_m( 0.f, 0.f,0.f,1.f); 
    glm::vec4 up_m(   0.f, 0.f,1.f,1.f); 
    glm::vec4 gze_m( look_m - eye_m ) ; 

    const glm::mat4& m2w = model2world ; 

    eye = m2w * eye_m ; 
    look = m2w * look_m  ; 
    up = m2w * up_m  ; 
    gaze = m2w * gze_m  ;    

    glm::vec3 forward_ax = glm::normalize(glm::vec3(gaze));
    glm::vec3 right_ax   = glm::normalize(glm::cross(forward_ax,glm::vec3(up))); 
    glm::vec3 top_ax     = glm::normalize(glm::cross(right_ax,forward_ax));

    float aspect = float(width)/float(height) ;
    float tanYfov = 1.f ;  // reciprocal of camera zoom
    float gazelength = glm::length( glm::vec3(gaze) ) ;
    float v_half_height = gazelength * tanYfov ;
    float u_half_width  = v_half_height * aspect ;

    U = glm::vec4( right_ax * u_half_width, 0.f) ;
    V = glm::vec4( top_ax * v_half_height,  0.f) ;
    W = glm::vec4( forward_ax * gazelength, 0.f) ; 
}

void CSGView::dump(const char* msg) const 
{
    std::cout << msg << std::endl ; 
    std::cout << desc("eye_model", eye_model ) << std::endl; 
    std::cout << desc("center_extent", center_extent ) << std::endl; 
    std::cout << desc("eye", eye ) << std::endl; 
    std::cout << desc("look", look ) << std::endl; 
    std::cout << desc("up", up ) << std::endl; 
    std::cout << desc("gaze", gaze ) << std::endl; 
    std::cout << desc("U", U ) << std::endl; 
    std::cout << desc("V", V ) << std::endl; 
    std::cout << desc("W", W ) << std::endl; 
}

std::string CSGView::desc( const char* label, const glm::vec4& v ) // static
{
    std::stringstream ss ; 
    ss 
       << std::setw(20) << label 
       << " ( "
       << std::setw(10) << std::fixed << std::setprecision(3) << v.x       
       << std::setw(10) << std::fixed << std::setprecision(3) << v.y
       << std::setw(10) << std::fixed << std::setprecision(3) << v.z       
       << std::setw(10) << std::fixed << std::setprecision(3) << v.w       
       << " ) "
       ;

    std::string s = ss.str();  
    return s ;
}

void CSGView::save(const char* dir) const 
{
    std::cout << "CSGView::save " << dir << std::endl ;  
    NP view("<f4", 9, 4); 
    float* f = view.values<float>() ; 

    std::vector<std::string> names ; 

    collect4(f,  0, eye_model);      names.push_back("eye_model");   
    collect4(f,  1, center_extent);  names.push_back("center_extent");
    collect4(f,  2, eye);            names.push_back("eye"); 
    collect4(f,  3, look);           names.push_back("look"); 
    collect4(f,  4, up);             names.push_back("up");
    collect4(f,  5, gaze);           names.push_back("gaze");
    collect4(f,  6, U);              names.push_back("U");
    collect4(f,  7, V);              names.push_back("V");
    collect4(f,  8, W);              names.push_back("W");
    view.save(dir, "view.npy"); 

    NP::WriteNames(dir, "view.txt", names); 
}

void CSGView::collect4( float* f, unsigned i, const glm::vec4& v ) // static
{
    *(f+4*i+0) = v.x ; 
    *(f+4*i+1) = v.y ; 
    *(f+4*i+2) = v.z ; 
    *(f+4*i+3) = v.w ; 
}


