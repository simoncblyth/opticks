#pragma once
/**
sdomain.h
===========


Regarding hc_eVnm see U4PhysicalConstantsTest, 
there is slight difference from smath.h float value : could be arising from CLHEP version difference

Wavelength fine domain np.linspace(60,820,761)

**/


#include <string>
#include <sstream>
#include <cassert>
#include <iomanip>

#include "NPFold.h"


struct sdomain
{
    static constexpr double hc_eVnm = 1239.84198433200208455673 ; 
    static constexpr const double    DOMAIN_LOW  = 60. ; 
    static constexpr const double    DOMAIN_HIGH = 820. ; 
    static constexpr const double    COARSE_DOMAIN_STEP = 20. ; 
    static constexpr const int       COARSE_DOMAIN_LENGTH = 39 ; 
    static constexpr const double    FINE_DOMAIN_STEP = 1. ; 
    static constexpr const int       FINE_DOMAIN_LENGTH = 761 ;  // 820-60+1   

    static constexpr const char     DOMAIN_TYPE = 'F' ;   // 'C'

    static constexpr double DomainLow(){    return DOMAIN_LOW ; }
    static constexpr double DomainHigh(){   return DOMAIN_HIGH ; }
    static constexpr double DomainStep(){   return DOMAIN_TYPE == 'F' ? FINE_DOMAIN_STEP    : COARSE_DOMAIN_STEP ; }
    static constexpr double DomainRange(){  return DOMAIN_HIGH - DOMAIN_LOW ; }

    static constexpr int    DomainLength(){  return DOMAIN_TYPE == 'F' ? FINE_DOMAIN_LENGTH : COARSE_DOMAIN_LENGTH ; }

    sdomain(); 

    NP* get_wavelength_nm() const ; 
    NP* get_energy_eV() const ; 
    NPFold* get_fold() const ; 

    static std::string Desc(const double* vv, int length, int edge); 
    std::string desc() const ; 

    int length ; 
    double step ; 
    double* wavelength_nm ; 
    double* energy_eV ;      // HMM: energy_eV  is descending following ascending wavelength_nm 
    double* spec4 ; 
};

inline sdomain::sdomain()
    :
    length(DomainLength()),
    step(DomainStep()),
    wavelength_nm(new double[length]),
    energy_eV(new double[length]),
    spec4(new double[4])
{
    for(int i=0 ; i < length ; i++) wavelength_nm[i] = DOMAIN_LOW + step*double(i) ; 
    assert( wavelength_nm[0] == DOMAIN_LOW ); 
    assert( wavelength_nm[length-1] == DOMAIN_HIGH ); 
    for(int i=0 ; i < length ; i++) energy_eV[i] = hc_eVnm/wavelength_nm[i] ; 

    spec4[0] = DomainLow(); 
    spec4[1] = DomainHigh();
    spec4[2] = DomainStep(); 
    spec4[3] = DomainRange(); 
}



inline NP* sdomain::get_wavelength_nm() const 
{
    return NP::MakeFromValues<double>( wavelength_nm, length ) ; 
}
inline NP* sdomain::get_energy_eV() const 
{
    return NP::MakeFromValues<double>( energy_eV, length ) ; 
}
inline NPFold* sdomain::get_fold() const 
{
    NPFold* fold = new NPFold ; 
    fold->add("wavelength_nm", get_wavelength_nm() ); 
    fold->add("energy_eV", get_energy_eV() ); 
    return fold ;  
}



inline std::string sdomain::Desc(const double* vv, int length, int edge) 
{
    std::stringstream ss ; 
    ss << "(" ; 
    for(int i=0 ; i < length ; i++) 
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
    int edge = 5 ; 
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

