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
#include <sstream>
#include <iomanip>
#include <limits>

#include "GVector.hh"

std::string gfloat3::desc() const 
{
    std::stringstream ss ; 
    ss
          << "(" << std::setw(10) << std::fixed << std::setprecision(3) << x 
          << " " << std::setw(10) << std::fixed << std::setprecision(3) << y 
          << " " << std::setw(10) << std::fixed << std::setprecision(3) << z
          << ")"
          ; 

    return ss.str(); 
}

std::string gfloat4::desc() const 
{
    std::stringstream ss ; 
    ss
          << "(" << std::setw(10) << std::fixed << std::setprecision(3) << x 
          << " " << std::setw(10) << std::fixed << std::setprecision(3) << y 
          << " " << std::setw(10) << std::fixed << std::setprecision(3) << z
          << " " << std::setw(10) << std::fixed << std::setprecision(3) << w
          << ")"
          ; 

    return ss.str(); 
}

std::string guint4::description() const 
{
    std::stringstream ss ; 
    unsigned umax = std::numeric_limits<unsigned>::max() ;


    ss << " (" ;

    if(x == umax) ss << "---" ;
    else          ss << std::setw(3) << x ;

    ss << "," ;

    if(y == umax) ss << "---" ;
    else          ss << std::setw(3) << y  ;
            
    ss << "," ;

    if(z == umax) ss << "---" ;
    else          ss << std::setw(3) << z  ;
 
    ss << "," ;

    if(w == umax) ss << "---" ;
    else          ss << std::setw(3) << w  ;
 

    ss << ")" ;


    return ss.str(); 
}



