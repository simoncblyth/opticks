#pragma once

union tquad
{
   float4 f ; 
   int4   i ; 
   uint4  u ; 
};

struct float4x4 
{
   float4 q0 ; 
   float4 q1 ; 
   float4 q2 ; 
   float4 q3 ; 
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

inline std::ostream& operator<<(std::ostream& os, const float4x4& v)
{
    tquad q3 ; 
    q3.f = v.q3 ; 

    os 
       << " 0f:" << v.q0 
       << " 1f:" << v.q1
       << " 2f:" << v.q2 
       << " 3u:" << q3.u
    ; 

    return os;
}





