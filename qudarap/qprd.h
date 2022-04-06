#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
   #define QPRD_METHOD __device__
#else
   #define QPRD_METHOD 
#endif 

/**
qprd.h
=========

NB: moving to quad2 instead of this : for easy persisting and mocking 

**/

struct qprd
{
    float3   normal ;
    float    t ;   
    unsigned identity ; 
    unsigned boundary ;   // fairly small number of distinct boundaries expected : 8 bits probably enough ( 0xff = 255 )
};


