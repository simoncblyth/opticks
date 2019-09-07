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


#pragma once

#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"

class BRAP_API BConverter {
  public:
     template<typename T, typename S> 
     static T round_to_even(const S& x) ;

     static short shortnorm( float v, float center, float extent ); 
     static unsigned char my__float2uint_rn( float fv ) ;
     static unsigned char my__float2uint_rn_kludge( float fv ) ;
    
     static short shortnorm_old( float v, float center, float extent ); 
     static unsigned char my__float2uint_rn_old( float fv ) ;


};

#include "BRAP_TAIL.hh"


