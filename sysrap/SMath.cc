#include <cassert>
#include <cmath>
#include <sstream>
#include <iomanip>

#include "SMath.hh"

/**
SMath::cos_pi
------------------

      
              Y
              |
              |
              |
    -X -------O--------- phi_pi = 0.   X
              |
              |
              |
             -Y

TODO: some platforms have sinpi cospi functions, could use those when available

**/

double SMath::cos_pi( double phi_pi )
{
    double cosPhi_0 = std::cos(phi_pi*M_PI) ; 
    double cosPhi_1 = cosPhi_0 ; 

    if( phi_pi == 0.0  ) cosPhi_1 = 1. ; 
    if( phi_pi == 0.5  ) cosPhi_1 = 0. ; 
    if( phi_pi == 1.0  ) cosPhi_1 = -1. ; 
    if( phi_pi == 1.5  ) cosPhi_1 = 0. ; 
    if( phi_pi == 2.0  ) cosPhi_1 = 1. ; 

    assert( std::abs(cosPhi_0 - cosPhi_1) < 1e-6 ) ; 
    return cosPhi_1 ; 
} 

double SMath::sin_pi( double phi_pi )
{
    double sinPhi_0 = std::sin(phi_pi*M_PI) ; 
    double sinPhi_1 = sinPhi_0 ; 

    if( phi_pi == 0.0  ) sinPhi_1 = 0. ; 
    if( phi_pi == 0.5  ) sinPhi_1 = 1. ; 
    if( phi_pi == 1.0  ) sinPhi_1 = 0. ; 
    if( phi_pi == 1.5  ) sinPhi_1 = -1. ; 
    if( phi_pi == 2.0  ) sinPhi_1 = 0. ; 

    assert( std::abs(sinPhi_0 - sinPhi_1) < 1e-6 ) ; 
    return sinPhi_1 ; 
} 


std::string SMath::Format( std::vector<std::pair<std::string, double>>& pairs, int l, int w, int p)
{
    std::stringstream ss ; 
    for(unsigned i=0 ; i < pairs.size() ; i++)
    {
        const std::pair<std::string, double>& pair = pairs[i] ; 
        ss
            << std::setw(3) << i << " " 
            << std::setw(l) << pair.first
            << std::scientific << std::setw(w) << std::setprecision(p) << pair.second  
            << std::endl
            ;
    }
    std::string s = ss.str(); 
    return s ; 
}



