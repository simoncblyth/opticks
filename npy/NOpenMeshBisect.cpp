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
#include <boost/math/tools/roots.hpp>

#include "NOpenMeshType.hpp"
//#include "NOpenMeshDesc.hpp"
#include "NOpenMeshBisect.hpp"


template <typename T>
NOpenMeshBisect<T>::NOpenMeshBisect( const SDF& sdf, const P& a, const P& b , float tolerance ) 
    :
    func(sdf, a, b ),
    tol(tolerance), 
    invalid( func(0)*func(1) > 0 ),  // <-- sdf must have different signs at a and b : ie bracket zero
    degenerate( fabs(func(0)) < tolerance && fabs(func(1)) <  tolerance ),  // <-- no root to find, already there 
    iterations(15)
{
}


template <typename T>
void NOpenMeshBisect<T>::bisect( P& frontier, float& t ) 
{
    std::pair<float, float> root = boost::math::tools::bisect(func, 0.f, 1.f, tol, iterations );
    
    t = (root.first + root.second)/2. ; 

    func.position(frontier, t );
}


template <typename T>
NOpenMeshBisectFunc<T>::NOpenMeshBisectFunc( const SDF& sdf, const P& a, const P& b ) 
    :
    sdf(sdf),
    a(a),
    b(b)
{
}
template <typename T>
void NOpenMeshBisectFunc<T>::position(P& tp, const float t) const 
{
    // parameterized position along a -> b line segment 
    //    a(1-t)+t*b   t=0 -> a,   t=1 -> b 

    const float s = 1.f - t ; 
    tp[0] = a[0]*s + b[0]*t ;
    tp[1] = a[1]*s + b[1]*t ;
    tp[2] = a[2]*s + b[2]*t ;
} 

template <typename T>
float NOpenMeshBisectFunc<T>::operator()(const float t) const 
{
    // signed distance to CSG left/right/composite object from position on line segement
    P tp ; 
    position(tp, t );

    float d = sdf( tp[0], tp[1], tp[2] ) ;

/*
    std::cout << "NOpenMeshBisectFunc<T>"
              << " t " << t 
              << " tp " << NOpenMeshDesc<T>::desc_point(tp,8,2)
              << " d " << d 
              << std::endl ; 
*/

    return d ; 
}


template struct NOpenMeshBisectFunc<NOpenMeshType> ;
template struct NOpenMeshBisect<NOpenMeshType> ;

