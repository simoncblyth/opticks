#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
   #define QPRD_METHOD __device__
#else
   #define QPRD_METHOD 
#endif 

/**
qprd.h
=========

**/

struct qprd
{
    float3   normal ;
    float    t ;   
    unsigned identity ; 
    unsigned boundary ; 
};


