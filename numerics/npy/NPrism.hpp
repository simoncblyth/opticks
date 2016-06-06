#pragma once
#include "NQuad.hpp"
#include <cmath>
#include <boost/math/constants/constants.hpp>

struct npart ;

struct nprism 
{
    nprism(float apex_angle_degrees=90.f, float height_mm=100.f, float depth_mm=100.f, float fallback_mm=100.f);
    nprism(const nvec4& param_);

    float height();
    float depth();
    float hwidth();

    npart part();
    void dump(const char* msg);

    nvec4 param ; 
};


inline nprism::nprism(float apex_angle_degrees, float height_mm, float depth_mm, float fallback_mm)
{
    param.x = apex_angle_degrees  ;
    param.y = height_mm  ;
    param.z = depth_mm  ;
    param.w = fallback_mm  ;
}

inline nprism::nprism(const nvec4& param_)
{
    param = param_ ;
}

inline float nprism::height()
{
    return param.y > 0.f ? param.y : param.w ; 
}
inline float nprism::depth()
{
    return param.z > 0.f ? param.z : param.w ; 
}
inline float nprism::hwidth()
{
    float pi = boost::math::constants::pi<float>() ;
    return height()*tan((pi/180.f)*param.x/2.0f) ;
}


