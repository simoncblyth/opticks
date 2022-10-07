#pragma once

#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <cassert>

#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"

struct U4PhysicalConstants
{
    static constexpr const char*  _hc_eVnm = "hc_eVnm" ; 
    static constexpr const double hc_eVnm = h_Planck*c_light/(eV*nm) ; 

    static constexpr const char*  _BirksConstant1_v0 = "BirksConstant1_v0" ; 
    static constexpr const double BirksConstant1_v0 = 12.05e3*g/cm2/MeV ;  // from $JUNOTOP/data/Simulation/DetSim/Material/LS/ConstantProperty 

    static constexpr const char*  _BirksConstant1_v1 = "BirksConstant1_v1" ; 
    static constexpr const double BirksConstant1_v1 = 12.05e-3*g/cm2/MeV ;  // suspected typo fix

    static constexpr const char*  _BirksConstant1_v2 = "BirksConstant1_v2" ; 
    static constexpr const double BirksConstant1_v2 = 0.0125*g/cm2/MeV ;    // from comment in code

    static constexpr const char*  _gval = "g" ; 
    static constexpr const double gval = g ; 

    static constexpr const char*  _gramval = "gram" ; 
    static constexpr const double gramval = gram ; 

    static constexpr const char*  _cm2val = "cm2" ; 
    static constexpr const double cm2val = cm2 ; 

    static constexpr const char*  _MeVval = "MeV" ; 
    static constexpr const double MeVval = MeV ; 

    static constexpr const char* _universe_mean_density = "universe_mean_density" ; 
    static constexpr const double universe_mean_density_ = universe_mean_density ; 

    static constexpr const char* _universe_mean_density_per = "universe_mean_density/(g/cm3)" ; 
    static constexpr const double universe_mean_density_per_ = universe_mean_density/(g/cm3) ; 

    static constexpr const char* _density_unit = "(g/cm3)" ; 
    static constexpr const double density_unit_ = (g/cm3) ; 

 


    static void Get(std::vector<std::string>& labels, std::vector<double>& values); 
    static std::string Desc();  
}; 

void U4PhysicalConstants::Get( std::vector<std::string>& labels, std::vector<double>& values) 
{
    labels.push_back( _hc_eVnm)                ;  values.push_back(hc_eVnm) ; 
    labels.push_back( _BirksConstant1_v0 )     ;  values.push_back(BirksConstant1_v0) ; 
    labels.push_back( _BirksConstant1_v1 )     ;  values.push_back(BirksConstant1_v1) ; 
    labels.push_back( _BirksConstant1_v2 )     ;  values.push_back(BirksConstant1_v2) ; 
    labels.push_back( _gval )                  ;  values.push_back(gval) ; 
    labels.push_back( _gramval )               ;  values.push_back(gramval) ; 
    labels.push_back( _cm2val )                ;  values.push_back(cm2val) ; 
    labels.push_back( _MeVval )                ;  values.push_back(MeVval) ; 
    labels.push_back( _universe_mean_density ) ;  values.push_back(universe_mean_density_) ; 
    labels.push_back( _universe_mean_density_per  ) ;  values.push_back(universe_mean_density_per_ ) ; 
    labels.push_back( _density_unit  )         ;  values.push_back(density_unit_ ) ; 
}

std::string U4PhysicalConstants::Desc()  // static
{
    std::vector<std::string> labels ; 
    std::vector<double> values ; 
    Get(labels, values); 
    assert( labels.size() == values.size() ); 
    std::stringstream ss ;

    for(unsigned i=0 ; i < values.size() ; i++)
    {
        const std::string& label = labels[i] ; 
        double value = values[i] ; 
        ss 
            << std::setw(20) << label 
            << " : "
            << std::fixed << std::setw(40) << std::setprecision(20) << value
            << " : "
            << std::setw(10) << std::scientific << value
            << std::endl 
            ;

    }
    std::string s = ss.str(); 
    return s ; 

}

