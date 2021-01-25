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

union tquad
{
   float4 f ; 
   int4   i ; 
   uint4  u ; 
};

struct cfloat6x4 
{
   float4 q0 ; 
   float4 q1 ; 
   float4 q2 ; 
   float4 q3 ; 
   float4 q4 ; 
   float4 q5 ; 
};
struct cfloat4x4 
{
   float4 q0 ; 
   float4 q1 ; 
   float4 q2 ; 
   float4 q3 ; 
};

struct cfloat2x4 
{
   float4 q0 ; 
   float4 q1 ; 
};


inline std::ostream& operator<<(std::ostream& os, const uint4& v)
{
    os << "(" << v.x << "," << v.y << "," << v.z << "," << v.w << ") " ; 
    return os;
}
inline std::ostream& operator<<(std::ostream& os, const float4& v)
{
    os << "(" << v.x << "," << v.y << "," << v.z << "," << v.w << ") " ; 
    return os;
}
inline std::ostream& operator<<(std::ostream& os, const cfloat6x4& v)
{
    tquad qlast ; 
    qlast.f = v.q5 ; 
    os 
       << " 0f:" << v.q0 
       << " 1f:" << v.q1
       << " 2f:" << v.q2 
       << " 3f:" << v.q3 
       << " 4f:" << v.q4 
       << " 5u:" << qlast.u
    ; 
    return os;
}
inline std::ostream& operator<<(std::ostream& os, const cfloat4x4& v)
{
    tquad qlast ; 
    qlast.f = v.q3 ; 
    os 
       << " 0f:" << v.q0 
       << " 1f:" << v.q1
       << " 2f:" << v.q2 
       << " 3u:" << qlast.u
    ; 

    return os;
}
inline std::ostream& operator<<(std::ostream& os, const cfloat2x4& v)
{
    tquad qlast ; 
    qlast.f = v.q1 ; 
    os 
       << " 0f:" << v.q0 
       << " 1u:" << qlast.u
    ; 
    return os;
}


