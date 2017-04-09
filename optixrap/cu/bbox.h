#pragma once

using namespace optix ; 




__device__ void transform_bbox(Aabb* aabb, const Matrix4x4* tr )
{
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


    rtPrintf("##test_tranBuffer irit\n%8.3f %8.3f %8.3f %8.3f\n%8.3f %8.3f %8.3f %8.3f\n%8.3f %8.3f %8.3f %8.3f\n", 
          irit[0], irit[1], irit[2], irit[3],  
          irit[4], irit[5], irit[6], irit[7],  
          irit[8], irit[9], irit[10], irit[11]
         );  
    rtPrintf("%8.3f %8.3f %8.3f %8.3f\n", irit[12], irit[13], irit[14], irit[15] );
}


__device__ void test_transform_bbox()
{
    float3 mn = make_float3(-100.f,-100.f,-100.f);
    float3 mx = make_float3( 100.f, 100.f, 100.f);

    Aabb bb(mn, mx);

    float angle = M_PIf*45.f/180.f ; 

    float3 ax = make_float3(0.f, 0.f, 1.f );
    float3 tl = make_float3(0.f, 0.f, 100.f );

    Matrix4x4 r = Matrix4x4::rotate( angle, ax ); 
    Matrix4x4 t = Matrix4x4::translate( tl ); 
    Matrix4x4 tr = t * r ; 

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




