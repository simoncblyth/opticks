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


template<typename T> 
struct ShortCompressor
{
    int _imax ; 
    T _center ; 
    T _extent ; 
    T _max    ; 
    T _step   ; 
    T _eps    ;
    T _half    ;

    ShortCompressor( T center, T extent )
        :
        _imax(32767),
        _center(center),
        _extent(extent),
        _max(_imax),
        _step(_extent/_max),
        _eps(0.001),
        _half(0.5)
    {   
    }   

    __host__ __device__ 
    T value(int iv) 
    {   
        return _center + _step*(T(iv)+_half) ;   
    }   

    __host__ __device__ 
    T fvalue(T v) 
    {   
        return  _max*(v - _center)/_extent ;
    }   
  
    __device__ 
    int ivalue(double v)
    {   
        T vv = _max*(v - _center)/_extent ;
        return __double2int_rn(vv)   ; 
    }   

    __device__ 
    int ivalue(float v)
    {   
        T vv = _max*(v - _center)/_extent ;
        return __float2int_rn(vv)   ; 
    }   


    __device__
    void operator()(float4 v)
    {
        printf("%15.7f %15.7f %d   %15.7f %d   %15.7f %d    %15.7f  %d \n",
              v.x, fvalue(v.x), ivalue(v.x), 
              v.y, ivalue(v.y),
              v.z, ivalue(v.z),
              v.w, ivalue(v.w)
        );
    }


    __host__
    void test(int d0)
    {
        thrust::device_vector<float4> fvec(10);
        // TODO: fix this, better to prep on host then copy to dev in one go
        for(int i=0 ; i < 10 ; i++)
        {
            T val = value(d0+i) ;
            fvec[i] = make_float4( val, val, val, T(d0+i) );
        }
        thrust::for_each(fvec.begin(), fvec.end(),  *this );
    }

        
  
};




int main()
{
    float center(0.);
    float extent(451.);

    ShortCompressor<float> comp(center,extent);
    comp.test(3445);

    cudaDeviceSynchronize();  

}


