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

//using namespace optix ; 

__device__ void transform_bbox(optix::Aabb* bb, const optix::Matrix4x4* tr )
{
   // http://dev.theomader.com/transform-bounding-boxes/
   // see: 
   //     npy-/NBBox.cpp 
   //     npy-/tests/NBBoxTest.cc

    float4 tr0 = tr->getRow(0) ;  
    float4 tr1 = tr->getRow(1) ;  
    float4 tr2 = tr->getRow(2) ;  
    float4 tr3 = tr->getRow(3) ;  

    float4 xa = tr0 * bb->m_min.x ; 
    float4 xb = tr0 * bb->m_max.x ;
    float4 xmi = fminf(xa, xb);
    float4 xma = fmaxf(xa, xb);

    float4 ya = tr1 * bb->m_min.y ; 
    float4 yb = tr1 * bb->m_max.y ;
    float4 ymi = fminf(ya, yb);
    float4 yma = fmaxf(ya, yb);

    float4 za = tr2 * bb->m_min.z ; 
    float4 zb = tr2 * bb->m_max.z ;
    float4 zmi = fminf(za, zb);
    float4 zma = fmaxf(za, zb);

    float4 tmi = xmi + ymi + zmi + tr3 ; 
    float4 tma = xma + yma + zma + tr3 ; 

    bb->m_min = make_float3( tmi.x, tmi.y, tmi.z );
    bb->m_max = make_float3( tma.x, tma.y, tma.z );
}


__device__ optix::Matrix4x4 make_test_matrix()
{
    float angle = M_PIf*45.f/180.f ; 

    float3 ax = make_float3(0.f, 0.f, 1.f );
    float3 tl = make_float3(0.f, 0.f, 100.f );

    optix::Matrix4x4 r = optix::Matrix4x4::rotate( angle, ax ); 
    optix::Matrix4x4 t = optix::Matrix4x4::translate( tl ); 
    optix::Matrix4x4 tr = t * r ; 

    return tr ; 
}



/*

// test_transform_bbox requires tranBuffer, use like this::
//
//     rtBuffer<optix::Matrix4x4> tranBuffer; 
//     #include "bbox.h"
//

__device__ void test_transform_bbox()
{
    int tranIdx = 0 ; 
    optix::Matrix4x4 tr = 2*tranIdx < tranBuffer.size() ? tranBuffer[2*tranIdx+0] : make_test_matrix() ; 

    unsigned size = tranBuffer.size() ;  // smth funny cannot directly rtPrintf a size
    rtPrintf("##test_transform_bbox tranBuffer size %u \n", size  );

    float3 mn = make_float3(-100.f,-100.f,-100.f);
    float3 mx = make_float3( 100.f, 100.f, 100.f);

    optix::Aabb bb(mn, mx);

    transform_bbox( &bb, &tr ) ; 

    rtPrintf("##test_transform_bbox tr\n%8.3f %8.3f %8.3f %8.3f\n%8.3f %8.3f %8.3f %8.3f\n%8.3f %8.3f %8.3f %8.3f\n", 
          tr[0], tr[1], tr[2], tr[3],  
          tr[4], tr[5], tr[6], tr[7],  
          tr[8], tr[9], tr[10], tr[11]
         );  
    rtPrintf("%8.3f %8.3f %8.3f %8.3f\n", tr[12], tr[13], tr[14], tr[15] );


    rtPrintf("##test_transform_bbox min %8.3f %8.3f %8.3f max %8.3f %8.3f %8.3f \n", 
         bb.m_min.x, bb.m_min.y, bb.m_min.z,
         bb.m_max.x, bb.m_max.y, bb.m_max.z);

}

*/


