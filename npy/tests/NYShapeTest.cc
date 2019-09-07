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
#include <iomanip>

#include "NYMath.hpp"
#include "NYShape.hpp"
#include "PLOG.hh"


typedef std::array<int,3>   int3 ;
typedef std::array<float,3> float3 ;
typedef std::array<float,2> float2 ;

struct ymesh
{
    std::vector<int3>   tri ;
    std::vector<float3> pos ;
    std::vector<float3> nrm ; 
    std::vector<float2> tex ;

    void dump(const char* msg="ymesh::dump") const ;
};

void ymesh::dump(const char* msg) const 
{
    std::cout << msg 
              << " n_tri " << tri.size()  
              << " n_pos " << pos.size()  
              << " n_nrm " << nrm.size()  
              << " n_tex " << tex.size()  
              << std::endl ; 

    for(auto t : tri)
    {
       int i = t[0] ; 
       int j = t[1] ; 
       int k = t[2] ; 

       std::cout << "t" 
                 << std::setw(4) << i 
                 << std::setw(4) << j 
                 << std::setw(4) << k
                 << "( "
                 << std::setw(15) << std::fixed << std::setprecision(2) << pos[i][0] 
                 << std::setw(15) << pos[i][1] 
                 << std::setw(15) << pos[i][2]
                 << ","
                 << std::setw(15) << pos[j][0] 
                 << std::setw(15) << pos[j][1] 
                 << std::setw(15) << pos[j][2]
                 << ","
                 << std::setw(15) << pos[k][0] 
                 << std::setw(15) << pos[k][1] 
                 << std::setw(15) << pos[k][2]
                 << ")"
                 << std::endl 
                 ;
    }
}


struct ysh
{
    virtual float3 par_pos(  const float2& uv ) const = 0 ;
    virtual float3 par_nrm( const float2& uv ) const = 0 ;
    virtual float2 par_tex( const float2& uv ) const = 0 ;
    ymesh*  make_mesh(int usteps, int vsteps) const ;
};

ymesh* ysh::make_mesh(int usteps, int vsteps) const 
{
    ymesh* m = new ymesh ; 

    std::function<float3(const float2&)> pos_fn  = [this](const float2& uv) { return par_pos(uv) ; } ;
    std::function<float3(const float2&)> nrm_fn  = [this](const float2& uv) { return par_nrm(uv) ; } ;
    std::function<float2(const float2&)> tex_fn  = [this](const float2& uv) { return par_tex(uv) ; } ;

    yshape::make_uvsurface( usteps, vsteps , m->tri, m->pos, m->nrm, m->tex, pos_fn, nrm_fn, tex_fn );
    return m ; 
}





struct ysphere : ysh
{
    ysphere(float radius) : radius(radius) {} ;

    float3 par_pos(  const float2& uv ) const ;
    float3 par_nrm( const float2& uv ) const ;
    float2 par_tex( const float2& uv ) const ;

    float radius ; 
};

float3 ysphere::par_pos( const float2& uv ) const 
{
    float phi = uv[0] * ym::pif;
    float theta = uv[1] * 2 * ym::pif ;
    float3 pos = {{ radius * cosf(theta) * sinf(phi), radius * sinf(theta) * sinf(phi), radius * cosf(phi) }} ;
    return pos ; 
}
float3 ysphere::par_nrm( const float2& uv ) const 
{
    float phi = uv[0] * ym::pif;
    float theta = uv[1] * 2 * ym::pif ;
    float3 pos = {{ cosf(theta) * sinf(phi), sinf(theta) * sinf(phi), cosf(phi) }} ;
    return pos ; 
}
float2 ysphere::par_tex( const float2& uv ) const 
{
    return uv ; 
}



void test_par()
{ 
    LOG(info) << "test_par" ; 

    ysphere sp(10) ; 
    float2 uv = {{0,0}} ;
    float3 pos = sp.par_pos(uv);
    std::cout << "pos( " << pos[0] << "," << pos[1] << "," << pos[2] << ")" << std::endl ;   
}

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    ysphere a(10) ; 
    ysphere b(1000) ; 

    ymesh* ma = a.make_mesh(8,8);
    ymesh* mb = b.make_mesh(8,8);

    ma->dump("a");
    mb->dump("b");

    return 0 ;   
}
