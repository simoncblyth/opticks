#pragma once

#include "NQuad.hpp"

struct npart ;

struct NPY_API nprism 
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



