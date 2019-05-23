#include <optix_world.h>
#include "quad.h"

using namespace optix;

rtBuffer<float4>   photon_buffer ;
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );
//rtDeclareVariable(unsigned int,  PNUMQUAD, , );   // in OptiX 501 510 this comes out zero 
rtDeclareVariable(uint2,  compaction_param, , );

//#define WITH_PRINT 1


RT_PROGRAM void compactionTest()
{
#ifdef WITH_PRINT
    unsigned photon_id = launch_index.x ;  
    unsigned photon_offset = photon_id*compaction_param.x ; 

    //rtPrintf("//compactionTest.cu photon_id %u photon_offset %u \n", photon_id, photon_offset  );

    union quad q0,q1,q2,q3 ;

    q0.f = photon_buffer[photon_offset+0] ;   
    q1.f = photon_buffer[photon_offset+1] ;   
    q2.f = photon_buffer[photon_offset+2] ;   
    q3.f = photon_buffer[photon_offset+3] ;   

    rtPrintf("compactionTest.cu  %5u %5u fffu(%10f, %10f, %10f, %u) \n", photon_id, photon_offset, q0.f.x, q1.f.y, q2.f.z, q3.u.w );
#endif
}

RT_PROGRAM void exception()
{
    rtPrintExceptionDetails();
}



