#pragma once

#include <string>
#include <sstream>
#include <iomanip>


#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"

struct U4PhysicalConstants
{
    static constexpr const double hc_eVnm = h_Planck*c_light/(eV*nm) ; 
    static std::string Desc();  
}; 

std::string U4PhysicalConstants::Desc()  // static
{
    std::stringstream ss ;
    ss 
        << std::setw(10) << "hc_eVnm"
        << " : "
        << std::fixed << std::setw(40) << std::setprecision(20) << hc_eVnm 
        << std::endl 
        ;
    std::string s = ss.str(); 
    return s ; 

}

