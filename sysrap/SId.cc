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
#include <cstring>
#include <cassert>

#include "SId.hh"

SId::SId(const char* identifiers_ )
   :
   identifiers(strdup(identifiers_)),
   len(strlen(identifiers)),
   idx(-1),
   cycle(0)
{
}

void SId::reset()
{
    idx = -1 ; 
    cycle = 0 ; 
}

const char* SId::get(bool reset_)
{
    if(reset_) reset();  

    if( idx + 1 == len ) 
    { 
        cycle += 1 ; 
        idx = -1 ; 
    }

    idx += 1 ; 
    assert( idx < len ) ; 

    std::stringstream ss ; 
    ss << identifiers[idx] ;
    if( cycle > 0 ) ss << cycle ; 
         
    std::string s = ss.str() ; 

    return strdup(s.c_str()) ; 
}


