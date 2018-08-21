#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>

using namespace optix;

//#include "numquad.h"

//rtDeclareVariable(float,         SPEED_OF_LIGHT, , );
rtDeclareVariable(unsigned int,  PNUMQUAD, , );
rtDeclareVariable(unsigned int,  RNUMQUAD, , );
rtDeclareVariable(unsigned int,  GNUMQUAD, , );

#include "quad.h"

rtBuffer<float4>               genstep_buffer ;
rtBuffer<float4>               photon_buffer;
rtBuffer<short4>               record_buffer;     // 2 short4 take same space as 1 float4 quad
rtBuffer<unsigned long long>   sequence_buffer;   // unsigned long long, 8 bytes, 64 bits 

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );
rtDeclareVariable(unsigned int,  record_max, , );


RT_PROGRAM void OEventTest()
{
    unsigned long long photon_id = launch_index.x ;  
    unsigned int photon_offset = photon_id*PNUMQUAD ;   // 4
    //union quad phead ;
    //phead.f = photon_buffer[photon_offset+0] ;

    //unsigned int genstep_id = phead.u.x ; 
    //unsigned int genstep_offset = genstep_id*GNUMQUAD ; 
    //union quad ghead ; 
    //ghead.f = genstep_buffer[genstep_offset+0]; 

//    rtPrintf("(OEventTest) photon_id %d photon_offset %d genstep_id %d GNUMQUAD %d genstep_offset %d \n", photon_id, photon_offset, genstep_id, GNUMQUAD, genstep_offset  );


    unsigned int MAXREC = record_max ; 
    int slot_min = photon_id*MAXREC ; 

    int record_offset = 0 ; 
    for(int slot=0 ; slot < MAXREC ; slot++)
    {
         record_offset = (slot_min + slot)*RNUMQUAD ;
         record_buffer[record_offset+0] = make_short4(slot,slot,slot,slot) ;    // 4*int16 = 64 bits
         record_buffer[record_offset+1] = make_short4(slot,slot,slot,slot) ;    
    }  

    photon_buffer[photon_offset+0] = make_float4( 0.f , 0.f, 0.f, 0.f );
    photon_buffer[photon_offset+1] = make_float4( 1.f , 1.f, 1.f, 1.f );
    photon_buffer[photon_offset+2] = make_float4( 2.f , 2.f, 2.f, 2.f );
    photon_buffer[photon_offset+3] = make_float4( 3.f , 3.f, 3.f, 3.f );

    unsigned long long seqhis = 0ull ; 
    unsigned long long seqmat = 1ull ; 

    sequence_buffer[photon_id*2 + 0] = seqhis ; 
    sequence_buffer[photon_id*2 + 1] = seqmat ;  

}

RT_PROGRAM void exception()
{
    rtPrintExceptionDetails();
}



