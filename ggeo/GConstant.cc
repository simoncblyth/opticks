#include "GConstant.hh"
//
// extract from CLHEP as do not wish to depend on that 
//    /usr/local/env/g4/geant4.10.02/source/externals/clhep/include/CLHEP/Units/SystemOfUnits.h 
//    /usr/local/env/g4/geant4.10.02/source/externals/clhep/include/CLHEP/Units/PhysicalConstants.h
//
// TODO: move this to optickscore- as globally applicable 
//

const double GConstant::meter = 1000. ;          // mm is 1 
const double GConstant::second = 1.e+9 ;         // ns is 1 
const double GConstant::electronvolt = 1.e-6 ;   // MeV is 1 

const double GConstant::nanometer = 1.e-9 *GConstant::meter;
const double GConstant::e_SI = 1.602176487e-19;  // positron charge in coulomb

const double GConstant::joule = GConstant::electronvolt/GConstant::e_SI;
const double GConstant::h_Planck = 6.62606896e-34 * GConstant::joule*GConstant::second ;
const double GConstant::c_light = 2.99792458e+8 * GConstant::meter/GConstant::second ;


