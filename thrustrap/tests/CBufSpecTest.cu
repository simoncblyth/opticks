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

#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include "CBufSpec.hh"


struct printf_functor_i
{
  __host__ __device__
  void operator()(int x)
  {
    printf("%d\n", x); 
  }
};


struct printf_functor_f4
{
  __host__ __device__
  void operator()(float4 v)
  {
    printf("%10.4f %10.4f %10.4f %10.4f \n", v.x, v.y, v.z, v.w);
  }
};



template <typename T>
CBufSpec make_bufspec(const thrust::device_vector<T>& d_vec )
{
    const T* raw_ptr = thrust::raw_pointer_cast(d_vec.data());

    unsigned int size = d_vec.size() ;
    unsigned int nbytes =  size*sizeof(T) ;

    return CBufSpec( (void*)raw_ptr, size, nbytes );
}



int main()
{
    thrust::device_vector<int> ivec(3);
    ivec[0] = 0;  
    ivec[1] = 1;  
    ivec[2] = 2;
    thrust::for_each(ivec.begin(), ivec.end(), printf_functor_i());


    CBufSpec ibs = make_bufspec<int>(ivec);
    ibs.Summary("ibs"); 


    thrust::device_vector<float4> fvec(3);
    fvec[0] = make_float4( 1.f, 2.f, 3.f, 4.f );  
    fvec[1] = make_float4( 1.f, 2.f, 3.f, 4.f );  
    fvec[2] = make_float4( 1.f, 2.f, 3.f, 4.f );  

    thrust::for_each(fvec.begin(), fvec.end(), printf_functor_f4());


    CBufSpec fbs = make_bufspec<float4>(fvec);
    fbs.Summary("fbs"); 




    cudaDeviceSynchronize();  

    // Without the sync the process will typically terminate before 
    // any output stream gets pumped out to the terminal when 
    // iterating over device_ptr. 
    // Curiously that doesnt seem to happen with device_vector ? 
    // Maybe their dtors are delayed by the dumping
}
