#pragma once

using namespace optix ; 

__device__ void transform_bbox(Aabb* bb, const Matrix4x4* tr )
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


__device__ void test_tranBuffer()
{
    int tranIdx = 0 ; 
    Matrix4x4 tr = tranBuffer[2*tranIdx+0] ; 
    Matrix4x4 irit = tranBuffer[2*tranIdx+1] ; 

    rtPrintf("##test_tranBuffer tr\n%8.3f %8.3f %8.3f %8.3f\n%8.3f %8.3f %8.3f %8.3f\n%8.3f %8.3f %8.3f %8.3f\n", 
          tr[0], tr[1], tr[2], tr[3],  
          tr[4], tr[5], tr[6], tr[7],  
          tr[8], tr[9], tr[10], tr[11]
         );  
    rtPrintf("%8.3f %8.3f %8.3f %8.3f\n", tr[12], tr[13], tr[14], tr[15] );

    float4 tr0 = tr.getRow(0) ;  
    float4 tr1 = tr.getRow(1) ;  
    float4 tr2 = tr.getRow(2) ;  
    float4 tr3 = tr.getRow(3) ;  

    rtPrintf("tr0\n%8.3f %8.3f %8.3f %8.3f\n", tr0.x, tr0.y, tr0.z, tr0.w );
    rtPrintf("tr1\n%8.3f %8.3f %8.3f %8.3f\n", tr1.x, tr1.y, tr1.z, tr1.w );
    rtPrintf("tr2\n%8.3f %8.3f %8.3f %8.3f\n", tr2.x, tr2.y, tr2.z, tr2.w );
    rtPrintf("tr3\n%8.3f %8.3f %8.3f %8.3f\n", tr3.x, tr3.y, tr3.z, tr3.w );


    rtPrintf("##test_tranBuffer irit\n%8.3f %8.3f %8.3f %8.3f\n%8.3f %8.3f %8.3f %8.3f\n%8.3f %8.3f %8.3f %8.3f\n", 
          irit[0], irit[1], irit[2], irit[3],  
          irit[4], irit[5], irit[6], irit[7],  
          irit[8], irit[9], irit[10], irit[11]
         );  
    rtPrintf("%8.3f %8.3f %8.3f %8.3f\n", irit[12], irit[13], irit[14], irit[15] );
}



__device__ Matrix4x4 make_test_matrix()
{
    float angle = M_PIf*45.f/180.f ; 

    float3 ax = make_float3(0.f, 0.f, 1.f );
    float3 tl = make_float3(0.f, 0.f, 100.f );

    Matrix4x4 r = Matrix4x4::rotate( angle, ax ); 
    Matrix4x4 t = Matrix4x4::translate( tl ); 
    Matrix4x4 tr = t * r ; 

    return tr ; 
}


__device__ void test_transform_bbox()
{
    float3 mn = make_float3(-100.f,-100.f,-100.f);
    float3 mx = make_float3( 100.f, 100.f, 100.f);

    Aabb bb(mn, mx);

    //Matrix4x4 tr = make_test_matrix();

    int tranIdx = 0 ; 
    Matrix4x4 tr = tranBuffer[2*tranIdx+0] ; 

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




