/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

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

const double GConstant::hc_eVnm = GConstant::h_Planck*GConstant::c_light/(GConstant::nanometer*GConstant::electronvolt) ;



