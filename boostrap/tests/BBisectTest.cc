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

#include <iomanip>
#include <iostream>
#include <functional>
#include <boost/math/tools/roots.hpp>


template <class T>
struct fn
{
    T operator()(const T& x )
    {
        return (x-2.)*(x-5.) ; 
    }
};

template <class T>
struct tolerance
{
    bool operator()(const T& min, const T& max )
    {
        return (max - min) < 0.001 ;   
    }
};




int main()
{

   fn<float> f ; 
   tolerance<float> tol ; 

   float min = 1 ; 
   float max = 3 ; 

   std::pair<float, float> r = boost::math::tools::bisect(f, min, max, tol );

   std::cout 
      << " r " << std::setw(15) << std::fixed << std::setprecision(4) << r.first
      << " " << r.second 
      << std::endl 
      ;


    return 0 ; 
}
