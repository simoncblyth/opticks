#pragma once

#include <iostream>
#include <iomanip>

struct s4
{
    float x,y,z,w ;
    float* data() ;
    std::string desc() const ;
};

inline float* s4::data()
{
    return &x ;
}

inline std::string s4::desc() const
{
    std::stringstream ss ;
    ss
      << "[s4"
      << " " << std::setw(10) << std::fixed << std::setprecision(3) << x
      << " " << std::setw(10) << std::fixed << std::setprecision(3) << y
      << " " << std::setw(10) << std::fixed << std::setprecision(3) << z
      << " " << std::setw(10) << std::fixed << std::setprecision(3) << w
      << "]"
      ;
    std::string str = ss.str() ;
    return str ;
}
