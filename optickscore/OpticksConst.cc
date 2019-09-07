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

#include <sstream>

#include "OpticksConst.hh"


const char* OpticksConst::BNDIDX_NAME_  = "Boundary_Index" ;
const char* OpticksConst::SEQHIS_NAME_  = "History_Sequence" ;
const char* OpticksConst::SEQMAT_NAME_  = "Material_Sequence" ;


std::string OpticksConst::describeModifiers(unsigned int modifiers)
{
    std::stringstream ss ; 
    if(modifiers & e_shift)   ss << "shift " ; 
    if(modifiers & e_control) ss << "control " ; 
    if(modifiers & e_option)  ss << "option " ; 
    if(modifiers & e_command) ss << "command " ;
    return ss.str(); 
}
bool OpticksConst::isShift(unsigned int modifiers) { return 0 != (modifiers & e_shift) ; }
bool OpticksConst::isOption(unsigned int modifiers) { return 0 != (modifiers & e_option) ; }
bool OpticksConst::isShiftOption(unsigned int modifiers) { return isShift(modifiers) && isOption(modifiers) ; }
bool OpticksConst::isCommand(unsigned int modifiers) { return 0 != (modifiers & e_command) ; }
bool OpticksConst::isControl(unsigned int modifiers) { return 0 != (modifiers & e_control) ; }


const char OpticksConst::GEOCODE_ANALYTIC = 'A';
const char OpticksConst::GEOCODE_TRIANGULATED = 'T' ;
const char OpticksConst::GEOCODE_GEOMETRYTRIANGLES = 'G' ;
const char OpticksConst::GEOCODE_SKIP = 'K' ;


