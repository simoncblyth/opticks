#include "Params.h"

#ifndef __CUDACC__
#include <glm/glm.hpp>
#include <iostream>

void Params::setView(const glm::vec3& eye_, const glm::vec3& U_, const glm::vec3& V_, const glm::vec3& W_ )
{
    eye.x = eye_.x ;
    eye.y = eye_.y ;
    eye.z = eye_.z ;

    U.x = U_.x ; 
    U.y = U_.y ; 
    U.z = U_.z ; 

    V.x = V_.x ; 
    V.y = V_.y ; 
    V.z = V_.z ; 

    W.x = W_.x ; 
    W.y = W_.y ; 
    W.z = W_.z ; 
}

void Params::setView(const glm::vec4& eye_, const glm::vec4& U_, const glm::vec4& V_, const glm::vec4& W_)
{
    eye.x = eye_.x ;
    eye.y = eye_.y ;
    eye.z = eye_.z ;

    U.x = U_.x ; 
    U.y = U_.y ; 
    U.z = U_.z ; 

    V.x = V_.x ; 
    V.y = V_.y ; 
    V.z = V_.z ; 

    W.x = W_.x ; 
    W.y = W_.y ; 
    W.z = W_.z ; 
}

void Params::setCamera(float tmin_, float tmax_, unsigned cameratype_ )
{
    tmin = tmin_ ; 
    tmax = tmax_ ; 
    cameratype = cameratype_ ; 

    std::cout << "Params::setCamera"
              << " tmin " << tmin  
              << " tmax " << tmax
              << " cameratype " << cameratype
              << std::endl 
              ;  

}


Params::Params(int raygenmode_, unsigned width, unsigned height, unsigned depth)
{
    setRaygenMode(raygenmode_); 
    setSize(width, height, depth); 
}

void Params::setRaygenMode(int raygenmode_)
{
    raygenmode = raygenmode_ ; 
}

void Params::setSize(unsigned width_, unsigned height_, unsigned depth_ )
{
    width = width_ ;
    height = height_ ;
    depth = depth_ ;

    origin_x = width_ / 2;
    origin_y = height_ / 2;
}
#endif


