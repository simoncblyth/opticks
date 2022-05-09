#pragma once

/**
sscint.h : replace (but stay similar to) : npy/NStep.hpp optixrap/cu/scintillationstep.h  
============================================================================================

* FOLLOWING PATTERN OF storch.h and scerenkov.h

For now just implemnet for JUNO specific variant : collectGenstep_DsG4Scintillation_r4695

**/


#if defined(__CUDACC__) || defined(__CUDABE__)
   #define SSCINT_METHOD __device__
#else
   #define SSCINT_METHOD 
#endif 

#include "OpticksGenstep.h"
#include "OpticksPhoton.h"

#include "scurand.h"
#include "smath.h"


struct sscint
{
    // ctrl
    unsigned gentype ;  
    unsigned trackid ; 
    unsigned matline ; 
    unsigned numphoton ; 
 
    float3   pos ;  // formerly x0
    float    time ; // formerly t0 

    float3 DeltaPosition ;
    float  step_length ;

    int code; 
    float charge ;
    float weight ;
    float meanVelocity ; 

    int   scnt ; 
    float f41 ; 
    float f42 ; 
    float f43 ; 
     
    float ScintillationTime ; 
    float f51 ; 
    float f52 ; 
    float f53 ; 


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
   static void FillGenstep( quad6& gs_, unsigned genstep_id, unsigned numphoton_per_genstep ) ; 
#endif

}; 


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
inline void sscint::FillGenstep( quad6& gs_, unsigned genstep_id, unsigned numphoton_per_genstep )
{
    sscint& gs = (sscint&)gs_ ; 

    gs.gentype = OpticksGenstep_SCINTILLATION ; 
    gs.numphoton = numphoton_per_genstep  ;   

    // fabricate some values for the genstep

    gs.pos.x = 100.f ; 
    gs.pos.y = 100.f ; 
    gs.pos.z = 100.f ; 
    gs.time = 20.f ; 

    gs.DeltaPosition.x = 1000.f ; 
    gs.DeltaPosition.y = 1000.f ; 
    gs.DeltaPosition.z = 1000.f ; 
    gs.step_length = 1000.f ; 

    gs.code = 1 ; 
    gs.charge = 1.f ;
    gs.weight = 1.f ;
    gs.meanVelocity = 10.f ; 

    gs.scnt = 0 ; 
    gs.f41 = 0.f ;   
    gs.f42 = 0.f ;   
    gs.f43 = 0.f ;   

    gs.ScintillationTime = 10.f ;
    gs.f51 = 0.f ;
    gs.f52 = 0.f ;
    gs.f53 = 0.f ;
}

#endif

