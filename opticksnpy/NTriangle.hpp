#pragma once

#include "NGLM.hpp"
#include "NPY_API_EXPORT.hh"
#include "NPY_HEAD.hh"

struct NPY_API ntriangle 
{
    ntriangle(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c);
    ntriangle(float* ptr);
    void copyTo(float* ptr) const;
    void dump(const char* msg="ntriangle::dump");

    glm::vec3 p[3];
}; 

#include "NPY_TAIL.hh"


