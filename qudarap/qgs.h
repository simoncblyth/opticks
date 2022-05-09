#pragma once
#include "squad.h"

#if defined(__CUDACC__) || defined(__CUDABE__)
   #define QGS_METHOD __device__
#else
   #define QGS_METHOD 
#endif 


struct ST
{
    int Id    ;
    int ParentId ;
    int MaterialIndex  ; // MaterialLine is clearer
    int NumPhotons ;

    float3 x0 ;
    float  t0 ;

    float3 DeltaPosition ;
    float  step_length ;

}; 

struct CK0
{
    int   code; 
    float charge ;
    float weight ;
    float preVelocity ; 

    float BetaInverse ; 
    float Pmin ; 
    float Pmax ; 
    float maxCos ; 

    float maxSin2 ;
    float MeanNumberOfPhotons1 ; 
    float MeanNumberOfPhotons2 ; 
    float postVelocity ; 
};

struct CK1
{
    int   code; 
    float charge ;
    float weight ;
    float preVelocity ; 

    float BetaInverse ; 
    float Wmin ; 
    float Wmax ; 
    float maxCos ; 

    float maxSin2 ;
    float MeanNumberOfPhotons1 ; 
    float MeanNumberOfPhotons2 ; 
    float postVelocity ; 
};


struct SC0
{
    int   code; 
    float charge ;
    float weight ;
    float preVelocity ; 

    int   scnt ;
    float slowerRatio ;   
    float slowTimeConstant ;    
    float slowerTimeConstant ;

    float ScintillationTime ;
    float ScintillationIntegralMax ;
    float Other1 ;
    float Other2 ;
}; 

struct SC1
{
    int   code; 
    float charge ;
    float weight ;
    float midVelocity ; 

    int   scnt ;
    float f41 ;   
    float f42 ; 
    float f43 ;

    float ScintillationTime ;
    float f51 ;
    float f52 ;
    float f53 ;
}; 

struct GS
{
    ST st ; 
    union
    {
        CK0 ck0 ; 
        CK1 ck1 ; 
        SC0 sc0 ; 
        SC1 sc1 ; 
    };
};


struct QG
{
    union
    {
        quad6 q ; 
        GS    g ;  
    };

    QGS_METHOD void load(const quad6* src, unsigned id);
}; 


inline QGS_METHOD void QG::load(const quad6* src, unsigned id)
{
    // try just equate entire thing ? 

    const quad6& sgs = *(src+id) ; 
    q.q0.f = sgs.q0.f ; 
    q.q1.f = sgs.q1.f ; 
    q.q2.f = sgs.q2.f ; 
    q.q3.f = sgs.q3.f ; 
    q.q4.f = sgs.q4.f ; 
    q.q5.f = sgs.q5.f ; 
}

