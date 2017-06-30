#pragma once

#include "NPY_API_EXPORT.hh"
#include <vector>
#include <string>

#include "NGLM.hpp"
#include "NSDF.hpp"

struct nnode ; 
struct nmat4triple ;

/*
N : debugging structural nodes/SDF
======================================
*/

 
struct NPY_API N 
{
    N(const nnode* node, const nmat4triple* transform, float surface_epsilon=1e-6f );
    glm::uvec4 classify(const std::vector<glm::vec3>& qq, float epsilon, unsigned expect );
    void dump_points(const char* msg);
    std::string desc() const ;

    const nnode*           node ;
    const nmat4triple*     transform ; 

    NSDF                   nsdf ; 
    glm::uvec4             tots ;
    unsigned               num ; 
 
    std::vector<glm::vec3> model ; 
    std::vector<glm::vec3> local ; 

};



