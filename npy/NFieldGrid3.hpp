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

#include <cstddef>
#include "NPY_API_EXPORT.hh"
#include "NGLM.hpp"

template <typename FVec, typename IVec, int DIM> struct NField ; 
template <typename FVec, typename IVec, int DIM> struct NGrid ; 


template<typename FVec, typename IVec>
struct NPY_API NFieldGrid3  
{
    NFieldGrid3( NField<FVec,IVec,3>* field, NGrid<FVec,IVec,3>* grid, bool offset=false ) ;
    
    // grid coordinate to field value
    float value( const IVec& ijk ) const ;
    float value_f( const FVec& ijkf ) const ;

    // grid coordinate to world position
    FVec position( const IVec& ijk ) const ; 
    FVec position_f( const FVec& ijkf, bool debug=false ) const ;  
    //FVec position_f( float i, float j, float k, bool debug=false ) const ;  

    NField<FVec,IVec,3>* field ; 
    NGrid<FVec,IVec,3>*  grid ; 

    bool     offset ; 

};




