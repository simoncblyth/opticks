#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

struct ntriangle 
{
    ntriangle(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c);
    ntriangle(float* ptr);
    void copyTo(float* ptr) const;
    void dump(const char* msg="ntriangle::dump");

    glm::vec3 p[3];
}; 




inline ntriangle::ntriangle(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c) 
{
    p[0] = a ; 
    p[1] = b ; 
    p[2] = c ; 
}

inline ntriangle::ntriangle(float* ptr)
{
    p[0] = glm::make_vec3(ptr) ; 
    p[1] = glm::make_vec3(ptr+3) ; 
    p[2] = glm::make_vec3(ptr+6) ; 
}    

inline void ntriangle::copyTo(float* ptr) const 
{
    memcpy( ptr+0, glm::value_ptr(p[0]), sizeof(float)*3 );
    memcpy( ptr+3, glm::value_ptr(p[1]), sizeof(float)*3 );
    memcpy( ptr+6, glm::value_ptr(p[2]), sizeof(float)*3 );
}



