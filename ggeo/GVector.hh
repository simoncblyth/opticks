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

#include <cmath>
#include <climits>
#include <string>
#include <algorithm>
#include <cstdio>

//#include "NGLM.hpp"
//#include <glm/gtx/component_wise.hpp>
//#include "NBBox.hpp"

#include "NQuad.hpp"
#include "NGLM.hpp"
#include "GMatrix.hh"

#include "GGEO_API_EXPORT.hh"


struct GGEO_API gfloat2 
{
    gfloat2() : u(0.f), v(0.f) {} ;
    gfloat2(float _u, float _v) : u(_u), v(_v) {} ;
    gfloat2(const gfloat2& other ) : u(other.u), v(other.v)  {} ;

    void Summary(const char* msg)
    {
        printf("%s gfloat2 %10.3f %10.3f \n", msg, u, v);
    }

    float u,v ;
};


struct GGEO_API gfloat3 
{
    static gfloat3 minimum(const gfloat3& a, const gfloat3& b)
    {
        return gfloat3( fmin(a.x, b.x), fmin(a.y, b.y), fmin(a.z, b.z) );
    }

    static  gfloat3 maximum(const gfloat3& a, const gfloat3& b)
    {
        return gfloat3( fmax(a.x, b.x), fmax(a.y, b.y), fmax(a.z, b.z) );
    }
    gfloat3() : x(0.f), y(0.f), z(0.f) {} ;
    gfloat3(float _x) : x(_x), y(_x), z(_x) {} ;
    gfloat3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {} ;
    gfloat3(const gfloat3& other ) : x(other.x), y(other.y), z(other.z)  {} ;
    gfloat3(const nvec3& other ) : x(other.x), y(other.y), z(other.z)  {} ;

    //gfloat3(const glm::vec3& other ) : x(other.x), y(other.y), z(other.z)  {} ;
    //glm::vec3 as_vec3() const 
    //{
    //    return glm::vec3(x,y,z);
    //}
   
    bool operator==(const gfloat3& other) const 
    {
        return x == other.x && y == other.y && z == other.z   ;
    }

    gfloat3& operator += (const gfloat3& other)
    {
         x += other.x ; 
         y += other.y ; 
         z += other.z ; 
         return *this ;
    }

    gfloat3& operator -= (const gfloat3& other)
    {
         x -= other.x ; 
         y -= other.y ; 
         z -= other.z ; 
         return *this ;
    }


    gfloat3& operator *= (const GMatrixF& m)
    {
       float _x, _y, _z ;   
       _x = m.a1 * x + m.a2 * y + m.a3 * z + m.a4;
       _y = m.b1 * x + m.b2 * y + m.b3 * z + m.b4;
       _z = m.c1 * x + m.c2 * y + m.c3 * z + m.c4;

       x = _x ; 
       y = _y ; 
       z = _z ; 

       return *this ;
    }

    void Summary(const char* msg)
    {
        printf("%s gfloat3 %10.3f %10.3f %10.3f\n", msg, x, y, z);
    }


    std::string desc() const ;


    float x,y,z ;
};







struct GGEO_API gfloat4 
{
    gfloat4() : x(0.f), y(0.f), z(0.f), w(0.f) {} ;
    gfloat4(float _x, float _y, float _z, float _w) : x(_x), y(_y), z(_z), w(_w) {} ;
    gfloat4(gfloat3& v, float _w) : x(v.x), y(v.y), z(v.z), w(_w) {} ;
    gfloat4(const gfloat4& other ) : x(other.x), y(other.y), z(other.z), w(other.w) {} ;

    bool operator==(const gfloat4& other) const 
    {
        return x == other.x && y == other.y && z == other.z && w == other.w  ;
    }

    gfloat4& operator *= (const GMatrixF& m)
    {
       float _x, _y, _z, _w ;   
       _x = m.a1 * x + m.a2 * y + m.a3 * z + m.a4 * w;
       _y = m.b1 * x + m.b2 * y + m.b3 * z + m.b4 * w ;
       _z = m.c1 * x + m.c2 * y + m.c3 * z + m.c4 * w ;
       _w = m.d1 * x + m.d2 * y + m.d3 * z + m.d4 * w ;

       x = _x ; 
       y = _y ; 
       z = _z ; 
       w = _w ; 

       return *this ;
    }

    void Summary(const char* msg)
    {
        printf("%s gfloat4 %10.3f %10.3f %10.3f %10.3f\n", msg, x, y, z, w);
    }

    std::string description()
    {
        char s[128] ;
        snprintf(s, 128, "gfloat4 %10.3f %10.3f %10.3f %10.3f ", x, y, z, w);
        return s ; 
    }
    std::string desc() const ;


    float x,y,z,w ;
};






struct GGEO_API guint3 
{
    guint3() : x(0), y(0), z(0) {} ;
    guint3(unsigned int _x, unsigned int _y, unsigned int _z) : x(_x), y(_y), z(_z) {} ;

    unsigned int x,y,z ;
};



struct GGEO_API guint4
{
    guint4() : x(0), y(0), z(0), w(0) {} ;
    guint4(unsigned int _x, unsigned int _y, unsigned int _z, unsigned int _w) : x(_x), y(_y), z(_z), w(_w) {} ;

    glm::uvec4 as_vec() const ;


    unsigned int operator[](unsigned int index) const
    {
        switch(index)
        {  
           case 0:return x; break;
           case 1:return y; break;
           case 2:return z; break;
           case 3:return w; break;
        }
        return UINT_MAX ; 
    } 

    bool operator==(const guint4& other) const 
    {
       return 
           x == other.x && 
           y == other.y && 
           z == other.z && 
           w == other.w 
           ;
    }
    void Summary(const char* msg) const 
    {
        printf("%s : %10u %10u %10u %10u \n", msg, x, y, z, w);
    }

    std::string description() const ;


    unsigned int x,y,z,w ;
};



// complaints about ctors
//union gquad 
//{
//   gfloat4 f ;  
//   guint4  u ;  
//};




