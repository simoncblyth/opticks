#pragma once
/**





**/
struct ST
{
    int Id    ;
    int ParentId ;
    int MaterialIndex  ;
    int NumPhotons ;

    float3 x0 ;
    float  t0 ;

    float3 DeltaPosition ;
    float  step_length ;

    int code; 
    float charge ;
    float weight ;
    float preVelocity ; 
}; 

struct CK
{
    float BetaInverse ; 
    float Pmin ; 
    float Pmax ; 
    float maxCos ; 

    float maxSin2 ;
    float MeanNumberOfPhotons1 ; 
    float MeanNumberOfPhotons2 ; 
    float postVelocity ; 
};

struct SC0
{
    int scnt ;
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
    int scnt ;
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
        CK ck ; 
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
        //TS    t ; 
    };

    void load(const quad6* src, unsigned id);  
}; 

void QG::load(const quad6* src, unsigned id)
{
    const quad6& s = *(src+id) ; 
    q.q0.f = s.q0.f ; 
    q.q1.f = s.q1.f ; 
    q.q2.f = s.q2.f ; 
    q.q3.f = s.q3.f ; 
    q.q4.f = s.q4.f ; 
    q.q5.f = s.q5.f ; 
}


