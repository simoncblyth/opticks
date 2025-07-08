#pragma once

/**
sf4.h (formerly s4.h, renamed to avoid clash with S4.h on case compromised macOS)
===================================================================================

**/

#include <iostream>
#include <iomanip>

struct sf4
{
    float x,y,z,w ;
    float* data() ;
    std::string desc() const ;
};

inline float* sf4::data()
{
    return &x ;
}

inline std::string sf4::desc() const
{
    std::stringstream ss ;
    ss
      << "[sf4"
      << " " << std::setw(10) << std::fixed << std::setprecision(3) << x
      << " " << std::setw(10) << std::fixed << std::setprecision(3) << y
      << " " << std::setw(10) << std::fixed << std::setprecision(3) << z
      << " " << std::setw(10) << std::fixed << std::setprecision(3) << w
      << "]"
      ;
    std::string str = ss.str() ;
    return str ;
}
