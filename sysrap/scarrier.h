#pragma once


#if defined(__CUDACC__) || defined(__CUDABE__)
   #define SCARRIER_METHOD __device__
#else
   #define SCARRIER_METHOD 
#endif 

#include "OpticksGenstep.h"
#include "OpticksPhoton.h"

#include "smath.h"
#include "scuda.h"
#include "squad.h"

struct scarrier
{
#if defined(__CUDACC__) || defined(__CUDABE__)
#else
   static void FillGenstep( quad6& gs, unsigned numphoton_per_genstep ) ; 
#endif

}; 


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
void scarrier::FillGenstep( quad6& gs, unsigned numphoton_per_genstep ) 
{
    gs.q0.u = make_uint4( OpticksGenstep_PHOTON_CARRIER, 0u, 0u, numphoton_per_genstep );   
    gs.q1.u = make_uint4( 0u,0u,0u,0u );  
    gs.q2.f = make_float4( 0.f, 0.f, 0.f, 0.f );   // post
    gs.q3.f = make_float4( 1.f, 0.f, 0.f, 1.f );   // dirw
    gs.q4.f = make_float4( 0.f, 1.f, 0.f, 500.f ); // polw
    gs.q5.f = make_float4( 0.f, 0.f, 0.f, 0.f );   // flag 
}
#endif




