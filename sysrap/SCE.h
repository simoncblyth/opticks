#pragma once

#include <glm/glm.hpp>
#include "scuda.h"
#include <vector>

struct SCE
{
    template<typename T>
    static void Corners(std::vector<glm::tvec4<T>>& corners, const glm::tvec4<T>& _ce ); 
    static void Corners(std::vector<float4>& corners, const float4& _ce ); 

    template<typename T>
    static void Midface(std::vector<glm::tvec4<T>>& midface, const glm::tvec4<T>& _ce ); 
    static void Midface(std::vector<float4>& midface, const float4& _ce ); 

};

/**
SCE::Corners
-------------

::

     ZYX 
   0:000    
   1:001    +X
   2:010    +Y
   3:011
   4:100    +Z
   5:101
   6:110
   7:111


               110----------111         
                |            |
                |            |
  +Z   100----------101      | 
        |       |    |       | 
        |       |    |       |
        |      010---|------011       +Y
        |            | 
        |            | 
  -Z   000----------001        -Y        
                
       -X           +X



**/

template<typename T>
inline void SCE::Corners(std::vector<glm::tvec4<T>>& corners, const glm::tvec4<T>& _ce ) // static
{
    for(int c=0 ; c < 8 ; c++) corners.push_back({  
               _ce.x + ( c & 1 ? _ce.w : -_ce.w ),
               _ce.y + ( c & 2 ? _ce.w : -_ce.w ),
               _ce.z + ( c & 4 ? _ce.w : -_ce.w ),
               T(1)
            }) ;
}

inline void SCE::Corners(std::vector<float4>& corners, const float4& _ce ) // static
{
    typedef float T ; 
    for(int c=0 ; c < 8 ; c++) corners.push_back({  
               _ce.x + ( c & 1 ? _ce.w : -_ce.w ),
               _ce.y + ( c & 2 ? _ce.w : -_ce.w ),
               _ce.z + ( c & 4 ? _ce.w : -_ce.w ),
               T(1)
            }) ;
}


template<typename T>
inline void SCE::Midface(std::vector<glm::tvec4<T>>& midface, const glm::tvec4<T>& _ce ) // static
{
    for(int i=0 ; i < 3 ; i++) for(int j=0 ; j < 2 ; j++)
    {
        T sign = ( j == 0 ? T(-1) : T(1) ) ; 
        midface.push_back({
                                 _ce.x + ( i == 0 ? sign*_ce.w : T(0) ), 
                                 _ce.y + ( i == 1 ? sign*_ce.w : T(0) ),   
                                 _ce.z + ( i == 2 ? sign*_ce.w : T(0) ),
                                 T(1)
                              });
        
    }
    midface.push_back({ _ce.x, _ce.y, _ce.z, T(1) });   
}



inline void SCE::Midface(std::vector<float4>& midface, const float4& _ce ) // static
{
    typedef float T ; 
    for(int i=0 ; i < 3 ; i++) for(int j=0 ; j < 2 ; j++)
    {
        T sign = ( j == 0 ? T(-1) : T(1) ) ; 
        midface.push_back({
                                 _ce.x + ( i == 0 ? sign*_ce.w : T(0) ), 
                                 _ce.y + ( i == 1 ? sign*_ce.w : T(0) ),   
                                 _ce.z + ( i == 2 ? sign*_ce.w : T(0) ),
                                 T(1)
                              });
        
    }
    midface.push_back({ _ce.x, _ce.y, _ce.z, T(1) });   
}


