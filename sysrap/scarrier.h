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
    quad q0 ; 
    quad q1 ; 
    quad q2 ; 
    quad q3 ; 
    quad q4 ; 
    quad q5 ; 

#if defined(__CUDACC__) || defined(__CUDABE__) || defined(MOCK_CURAND) 
   SCARRIER_METHOD static void generate( sphoton& p, curandStateXORWOW& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id ); 
#endif

#if defined(__CUDACC__) || defined(__CUDABE__) || defined(MOCK_CURAND) 
#else
   SCARRIER_METHOD static void FillGenstep( scarrier& gs, unsigned matline, unsigned numphoton_per_genstep ) ; 
#endif

}; 


#if defined(__CUDACC__) || defined(__CUDABE__) || defined(MOCK_CURAND) 
inline SCARRIER_METHOD void scarrier::generate( sphoton& p_, curandStateXORWOW& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id )  // static
{
    quad4& p = (quad4&)p_ ; 

    p.q0.f = gs.q2.f ; 
    p.q0.f.y += float(photon_id)*10.f ; 

    p.q1.f = gs.q3.f ; 
    p.q2.f = gs.q4.f ; 
    p.q3.f = gs.q5.f ; 
    p.set_flag(TORCH); 
}
#endif


#if defined(__CUDACC__) || defined(__CUDABE__) || defined(MOCK_CURAND) 
#else
inline void scarrier::FillGenstep( scarrier& gs, unsigned matline, unsigned numphoton_per_genstep ) 
{
    gs.q0.u = make_uint4( OpticksGenstep_CARRIER, 0u, 0u, numphoton_per_genstep );   
    gs.q1.u = make_uint4( 0u,0u,0u,0u );  
    gs.q2.f = make_float4( 0.f, 0.f, 0.f, 0.f );   // post
    gs.q3.f = make_float4( 1.f, 0.f, 0.f, 1.f );   // dirw
    gs.q4.f = make_float4( 0.f, 1.f, 0.f, 500.f ); // polw
    gs.q5.f = make_float4( 0.f, 0.f, 0.f, 0.f );   // flag

    gs.q3.u.w = 0u ; // former weight from "dirw" is now being used for prd.iindex
 
}
#endif



