#pragma once
/**
U4SurfacePerfect.h
===================

Used for only for dummy testing surfaces with constant properties.
This is not used for real surfaces which have properties that 
vary with energy/wavelength.  

**/

#include <vector>
#include <string>
#include <sstream>
#include <iomanip>

struct U4SurfacePerfect
{
    static void Get(std::vector<U4SurfacePerfect>& perfects) ; 
    std::string name ; 
    double  detect ; 
    double  absorb ; 
    double  reflect_specular ; 
    double  reflect_diffuse ; 

    double sum() const ; 
    std::string desc() const ; 
}; 

inline double U4SurfacePerfect::sum() const 
{
    return detect + absorb + reflect_specular + reflect_diffuse ; 
}

inline std::string U4SurfacePerfect::desc() const 
{
    std::stringstream ss ; 
    ss << " detect " << std::setw(10) << std::fixed << std::setprecision(3) << detect ; 
    ss << " absorb " << std::setw(10) << std::fixed << std::setprecision(3) << absorb ; 
    ss << " reflect_specular " << std::setw(10) << std::fixed << std::setprecision(3) << reflect_specular ; 
    ss << " reflect_diffuse " << std::setw(10) << std::fixed << std::setprecision(3) << reflect_diffuse ; 
    ss << " name " << name ; 
    std::string str = ss.str(); 
    return str ; 
}


inline void U4SurfacePerfect::Get(std::vector<U4SurfacePerfect>& perfects) // static
{
    U4SurfacePerfect perfectDetectSurface   = { "perfectDetectSurface",   1., 0., 0., 0. } ; 
    U4SurfacePerfect perfectAbsorbSurface   = { "perfectAbsorbSurface",   0., 1., 0., 0. } ; 
    U4SurfacePerfect perfectSpecularSurface = { "perfectSpecularSurface", 0., 0., 1., 0. } ; 
    U4SurfacePerfect perfectDiffuseSurface  = { "perfectDiffuseSurface",  0., 0., 0., 1. } ; 

    perfects.push_back(perfectDetectSurface) ; 
    perfects.push_back(perfectAbsorbSurface) ; 
    perfects.push_back(perfectSpecularSurface) ; 
    perfects.push_back(perfectDiffuseSurface) ; 
}

