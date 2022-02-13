#pragma once

#include <vector>
#include <string>
#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SMath 
{
    static double cos_pi( double phi_pi ) ; 
    static double sin_pi( double phi_pi ) ; 

    static std::string Format( std::vector<std::pair<std::string, double>>& pairs, int l=15, int w=20, int p=10); 

}; 


