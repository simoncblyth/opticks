#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
   #define QSCINT_METHOD __device__
#else
   #define QSCINT_METHOD 
#endif 

struct qscint
{
    // gs0
    int    Id ; 
    int    ParentId ; 
    int    Spare0 ; 
    int    NumPhotons ; 

    // gs1
    float3 X0 ; 
    float  T0 ; 

    // gs2 
    float3 DeltaPosition ; 
    float  StepLength ;

    // gs3
    int   Code ; 
    float Charge ;
    float Spare1 ;  
    float MidVelocity ; 

    // gs4
    float Spare2 ; 
    float Spare3 ; 
    float Spare4 ; 
    float Spare5 ; 
 
    // gs5 
    float ScintillationTime ; 
    float Spare6 ; 
    float Spare7 ; 
    float Spare8 ; 


    QSCINT_METHOD void  fabricate_genstep();

};


QSCINT_METHOD void qscint::fabricate_genstep()
{
    X0 = make_float3( 1000.f, 1000.f, 1000.f ); 
    T0 = 0.f ; 

    DeltaPosition = make_float3( 1.f, 1.f, 1.f ); 
    StepLength = 100.f ;

    Charge = 1.f ; 
    MidVelocity = 300.f ; 

    ScintillationTime = 100.f ; 
}


