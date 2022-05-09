#pragma once
/**
scerenkov.h : replace (but stay similar to) : npy/NStep.hpp optixrap/cu/cerenkovstep.h  
========================================================================================

* FOLLOWING PATTERN OF storch.h 

**/


#if defined(__CUDACC__) || defined(__CUDABE__)
   #define SCERENKOV_METHOD __device__
#else
   #define SCERENKOV_METHOD 
#endif 

#include "OpticksGenstep.h"
#include "OpticksPhoton.h"

#include "scurand.h"
#include "smath.h"

// HMM could have scerenkovtype.h (like scerenkovtype.h) if need to handle different versions 


struct scerenkov
{
    //int Id    ;   
    //int ParentId ;
    //int MaterialIndex  ;
    //int NumPhotons ;

    // ctrl
    unsigned gentype ;  
    unsigned trackid ; 
    unsigned matline ; 
    unsigned numphoton ; 
 
    //float3 x0 ;
    //float  t0 ;
    float3   pos ;
    float    time ; 

    float3 DeltaPosition ;
    float  step_length ;

    int code; 
    float charge ;
    float weight ;
    float preVelocity ; 

    /// the above first 4 quads are common to both CerenkovStep and ScintillationStep 

    float BetaInverse ; 
    float Pmin ;   //   misleadingly this may be Wmin see G4Opticks::collectGenstep_G4Cerenkov_1042
    float Pmax ;   //   misleadingly this may be Wmax see G4Opticks::collectGenstep_G4Cerenkov_1042  
    float maxCos ; 
 
    float maxSin2 ;
    float MeanNumberOfPhotons1 ; 
    float MeanNumberOfPhotons2 ; 
    float postVelocity ; 

    // above are loaded parameters, below are derived from them
    //float MeanNumberOfPhotonsMax ; 
    //float3 p0 ;
    // NB : organized into 6 quads : are constained not to change that 

#if defined(__CUDACC__) || defined(__CUDABE__) || defined(MOCK_CURAND) 
   SCERENKOV_METHOD static void generate( sphoton& p, curandStateXORWOW& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id ); 
#endif

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
   static void FillGenstep( scerenkov& gs, unsigned genstep_id, unsigned numphoton_per_genstep ) ; 
   std::string desc() const ; 
#endif


};

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
inline void scerenkov::FillGenstep( scerenkov& gs, unsigned genstep_id, unsigned numphoton_per_genstep )
{
    gs.gentype = OpticksGenstep_CERENKOV ; 
    gs.numphoton = numphoton_per_genstep  ;   
}

inline std::string scerenkov::desc() const 
{
    std::stringstream ss ; 
    ss << "scerenkov::desc"
       << " gentype " << gentype 
       ;
    std::string s = ss.str(); 
    return s ; 
} 
#endif


#if defined(__CUDACC__) || defined(__CUDABE__) || defined(MOCK_CURAND) 

SCERENKOV_METHOD void scerenkov::generate( sphoton& p, curandStateXORWOW& rng, const quad6& gs_, unsigned photon_id, unsigned genstep_id )  // static
{
    const scerenkov& gs = (const scerenkov&)gs_ ;   // casting between union-ed types  

#ifdef SCERENKOV_DEBUG
    printf("//scerenkov::generate photon_id %3d genstep_id %3d  gs gentype/trackid/matline/numphoton(%3d %3d %3d %3d) type %d \n", 
       photon_id, 
       genstep_id, 
       gs.gentype, 
       gs.trackid,
       gs.matline, 
       gs.numphoton
      );  
#endif

}

#endif



/**
* qcerenkov : union between quad6 and specific genstep types for easy usage and yet no serialize/deserialize needed
**/

union qcerenkov
{
   quad6     q ; 
   scerenkov c ; 
};   


