#pragma once
/**
CSGView.h
==========

glm based eye, look, up projection transform maths

* HMM: no CSG dependency, this can and should be done at lower sysrap level
* TODO: review users and check for duplicated functionality   

**/

#include <string>
#include <glm/glm.hpp>

struct CSGView
{
    glm::vec4 eye_model ; 
    glm::vec4 center_extent ; 
    glm::vec4 eye ; 
    glm::vec4 look ; 
    glm::vec4 up ; 
    glm::vec4 gaze ; 
    glm::vec4 U ; 
    glm::vec4 V ; 
    glm::vec4 W ; 

    void update(const glm::vec4& eye_model, const glm::vec4& ce, const unsigned width, const unsigned height ); 
    void dump(const char* msg="CSGView::dump") const ;
    void save(const char* dir) const ;

    static std::string desc( const char* label, const glm::vec4& v );
    static void collect4( float* f, unsigned i, const glm::vec4& v ); // static
};


