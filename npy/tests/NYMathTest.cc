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

#include <iostream>
#include <iomanip>

#include "NYMath.hpp"
#include "PLOG.hh"


int main(int argc, char** argv)
{
    PLOG_(argc, argv);


    std::cout << std::setw(15) << " ym::pow2(10) " << ym::pow2(10) << std::endl ; 
    std::cout << std::setw(15) << " ym::pif " << ym::pif << std::endl ; 
    std::cout << std::setw(15) << " ym::pi " << ym::pi << std::endl ; 

    ym::vec4f v{1.,2.,3.,4.} ; 

    std::cout << " v.x " << v.x << " v[0] " << v[0] << std::endl ;  
    std::cout << " v.y " << v.y << " v[1] " << v[1] << std::endl ;  
    std::cout << " v.z " << v.z << " v[2] " << v[2] << std::endl ;  
    std::cout << " v.w " << v.w << " v[3] " << v[3] << std::endl ;  


    ym::vec4f v0{ 0., 1., 2., 3.} ; 
    ym::vec4f v1{10.,11.,12.,13.} ; 
    ym::vec4f v2{20.,21.,22.,23.} ; 
    ym::vec4f v3{30.,31.,32.,33.} ; 

    ym::mat4f m{v0,v1,v2,v3};

    assert( m.w.w == 33. );
    assert( m.w.z == 32. );
    assert( m.z.w == 23. );

    std::cout << " m.w.w " << m.w.w << " m[3][3] " << m[3][3] << std::endl ; 
    std::cout << " m.w.z " << m.w.z << " m[3][2] " << m[3][2] << std::endl ; 
    std::cout << " m.z.w " << m.z.w << " m[2][3] " << m[2][3] << std::endl ; 


    std::array<std::array<float, 3>, 4> frame = {{ {{0, 1, 2}}, {{10, 11, 12}}, {{20, 21, 22}}, {{30, 31, 32}} }} ;

    std::cout << " frame[0][0] " << frame[0][0] << std::endl; 
    std::cout << " frame[2][3] " << frame[2][3] << std::endl; 
    std::cout << " frame[3][0] " << frame[3][0] << std::endl; 
    assert( frame[2][3] == 30 );  // no range check : the 3 is out of range but this just steps along and accesses the next 
    assert( frame[3][0] == 30 ); 

    std::cout << " frame[3][2] " << frame[3][2] << std::endl; 
    assert( frame[3][2] == 32 );




    return 0 ;   
}
