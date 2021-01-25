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

#include "float4x4.h"

/**
TIsHit functor
===============

Decision based on all bits in the selection mask being set in photon flags, 
note this is non-exclusive, other bits may be set too.
Canonically only a single bit is set in the mask with SURFACE_DETECT.  

* photon flags such as SURFACE_DETECT are assigned to s.flag by oxrap-/cu/propagate.h
* at each bounce oxrap-/cu/generate.cu FLAGS macro ORs s.flag into p.flags.u.w
* p.flags.f saved into photon buffer by oxrap-/cu/photon.h:psave


Suspect a failure to rebuild dependency issue regards 
updates to this header... so touch the principal user TBuf\_.cu 
when changing this as workaround.

Potentially a template instanciation and nvcc issue ? 

It may be easier to surface this functor type as a template 
argument.


TODO: get rid of the 4x4 and 2x4 if the templated one works OK


**/

struct TIsHit4x4 : public thrust::unary_function<float4x4,bool>
{
    unsigned hitmask ; 

    TIsHit4x4(unsigned hitmask_) : hitmask(hitmask_) {}   

    __host__ __device__
    bool operator()(float4x4 photon)
    {   
        tquad q3 ; 
        q3.f = photon.q3 ; 
        return ( q3.u.w & hitmask ) == hitmask ;   
    }   
};


struct TIsHit2x4 : public thrust::unary_function<float2x4,bool>
{
    unsigned hitmask ; 

    TIsHit2x4(unsigned hitmask_) : hitmask(hitmask_) {}   

    __host__ __device__
    bool operator()(float2x4 way)
    {   
        tquad q1 ; 
        q1.f = way.q1 ; 
        return ( q1.u.w & hitmask ) == hitmask ;   
    }   
};


template <typename T>
struct TIsHit : public thrust::unary_function<T,bool>
{
    unsigned hitmask ; 

    TIsHit(unsigned hitmask_) : hitmask(hitmask_) {}   

    __host__ __device__
    bool operator()(T item)
    {   
        return ( item.flags() & hitmask ) == hitmask ;   
    }   
};





/**

python -i $(which evt.py --tag 10)

In [5]: flag = evt.ox[:,3,3].view(np.uint32)

In [6]: flag
Out[6]: 
A()sliced
A([6272, 6272, 6272, ..., 6272, 6208, 6272], dtype=uint32)

In [7]: count_unique(flag)
Out[7]: 
array([[  4104,  18866],
       [  4136,    197],
       [  4256,   3214],
       [  5128,    132],
       [  5248,   1446],
       [  5280,     45],
       [  6152,   3900],
       [  6176,      6],
       [  6184,     24],
       [  6208, 107599],  <<< SHOULD SELECT THESE
       [  6240,    113],  <<<  AND THESE
       [  6272, 363487],
       [  6304,    693],
       [  7168,      7],
       [  7176,     28],
       [  7200,      5],
       [  7208,      4],
       [  7296,    190],
       [  7328,     44]])

In [43]: TORCH | SURFACE_DETECT | BOUNDARY_TRANSMIT
Out[43]: 6208

n [47]: ( TORCH | SURFACE_DETECT | BOUNDARY_TRANSMIT | BULK_SCATTER  )
Out[47]: 6240


 **/

