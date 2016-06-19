
#include "GLMPrint.hpp"
#include "BLog.hh"

#include "NTriangle.hpp"

ntriangle::ntriangle(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c) 
{
    p[0] = a ; 
    p[1] = b ; 
    p[2] = c ; 
}

ntriangle::ntriangle(float* ptr)
{
    p[0] = glm::make_vec3(ptr) ; 
    p[1] = glm::make_vec3(ptr+3) ; 
    p[2] = glm::make_vec3(ptr+6) ; 
}    

void ntriangle::copyTo(float* ptr) const 
{
    memcpy( ptr+0, glm::value_ptr(p[0]), sizeof(float)*3 );
    memcpy( ptr+3, glm::value_ptr(p[1]), sizeof(float)*3 );
    memcpy( ptr+6, glm::value_ptr(p[2]), sizeof(float)*3 );
}

void ntriangle::dump(const char* msg)
{
    LOG(info) << msg ; 
    print(p[0], "p[0]");
    print(p[1], "p[1]");
    print(p[2], "p[2]");
}



