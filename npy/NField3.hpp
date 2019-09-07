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

#include <functional>
#include "NQuad.hpp"

#include "NPY_API_EXPORT.hh"



template<typename FVec, typename IVec, int DIM>
struct NPY_API NField
{
    static const int ZCORNER = 8 ; 
    static const FVec ZOFFSETS[ZCORNER] ;

    typedef std::function<float(float,float,float)> FN ;  
    NField( FN* f, const FVec& min, const FVec& max);
    std::string desc();

    FVec position( const FVec& fpos ) const;         // fractional position in 0:1 to world position in min:max

    float operator()( const FVec& fpos ) const;  // fractional position in 0:1 to field value

    int zcorners( const FVec& fpos, float fdelta ) const ;

    FN*  f ; 
    FVec min  ;
    FVec max  ;
    FVec side ; 


};


