#pragma once

#include <string>
#include <sstream>
#include <cassert>
#include <iomanip>

struct sdomain
{
    static constexpr double hc_eVnm = 1239.84198433200208455673 ; 
    // see U4PhysicalConstantsTest : there is slight difference from smath.h float value : could be arising from CLHEP version difference
    static constexpr const double    DOMAIN_LOW  = 60. ; 
    static constexpr const double    DOMAIN_HIGH = 820. ; 
    static constexpr const double    DOMAIN_STEP = 20. ; 
    static constexpr const unsigned DOMAIN_LENGTH = 39 ; 
    static constexpr const char     DOMAIN_TYPE = 'F' ;   // 'C'
    static constexpr const double   FINE_DOMAIN_STEP = 1. ; 
    static constexpr const unsigned FINE_DOMAIN_LENGTH = 761 ; 
 
    static constexpr double   DomainStep(){   return DOMAIN_TYPE == 'F' ? FINE_DOMAIN_STEP    : DOMAIN_STEP ; }
    static constexpr unsigned DomainLength(){  return DOMAIN_TYPE == 'F' ? FINE_DOMAIN_LENGTH : DOMAIN_LENGTH ; }

    sdomain(); 

    static std::string Desc(const double* vv, unsigned length, unsigned edge); 
    std::string desc() const ; 

    unsigned length ; 
    double step ; 
    double* wavelength_nm ; 
    double* energy_eV ; 
};

inline sdomain::sdomain()
    :
    length(DomainLength()),
    step(DomainStep()),
    wavelength_nm(new double[length]),
    energy_eV(new double[length])
{
    for(unsigned i=0 ; i < length ; i++) wavelength_nm[i] = DOMAIN_LOW + step*double(i) ; 
    assert( wavelength_nm[0] == DOMAIN_LOW ); 
    assert( wavelength_nm[length-1] == DOMAIN_HIGH ); 

    for(unsigned i=0 ; i < length ; i++) energy_eV[i] = hc_eVnm/wavelength_nm[i] ; 
}

inline std::string sdomain::Desc(const double* vv, unsigned length, unsigned edge) 
{
    std::stringstream ss ; 
    ss << "(" ; 
    for(unsigned i=0 ; i < length ; i++) 
    {
        if( i < edge || i > length - edge )
            ss << std::fixed << std::setw(10) << std::setprecision(5) << vv[i] << " " ; 
        else if( i == edge )
            ss << "... " ; 
    }
    ss << ")" ; 
    std::string s = ss.str(); 
    return s ; 
}
inline std::string sdomain::desc() const
{
    unsigned edge = 5 ; 
    std::stringstream ss ; 
    ss 
        << "sdomain::desc"
        << " length " << length  << std::endl 
        << " wavelength_nm " << Desc(wavelength_nm, length, edge) << std::endl 
        << " energy_eV     " << Desc(energy_eV,     length, edge) << std::endl 
        ;
    std::string s = ss.str(); 
    return s ; 
}

