#pragma once

#include <vector>
#include <functional>

#include "NPY_API_EXPORT.hh"
#include "NGLM.hpp"

struct NPY_API NSDF
{
   /*
   Applies inverse transform v to take global positions 
   into frame of the structural nd 
   where the CSG is placed, these positions local 
   to the CSG can then be used with the CSG SDF to 
   see the distance from the point to the surface of the 
   solid.
   */

    typedef std::vector<float>::const_iterator VFI ; 

    NSDF(std::function<float(float,float,float)> sdf, const glm::mat4& inverse );
    float operator()( const glm::vec3& q_ );

    void clear(); 
    void classify( const std::vector<glm::vec3>& qq, float epsilon, unsigned expect)  ;

    bool is_error() const ;
    bool is_empty() const ;
    std::string desc() const ;
    std::string detail() const ;


    std::function<float(float,float,float)> sdf ; 
    const glm::mat4                         inverse ; 

    std::vector<float>                      sd ; 
    glm::uvec4                              tot ; 
    glm::vec2                               range ;                  

    // hang on to last classification prameters for dumping 
    float    epsilon ; 
    unsigned expect ; 
    const std::vector<glm::vec3>*           qqptr ; 


};


