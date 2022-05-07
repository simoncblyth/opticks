#pragma once
/**
qdebug.h
==========

Instanciation managed from QSim

**/

#include "scuda.h"
#include "squad.h"
#include "sphoton.h"

#include "qstate.h"
#include "qprd.h"


#if defined(__CUDACC__) || defined(__CUDABE__)
   #define QDEBUG_METHOD __device__
#else
   #define QDEBUG_METHOD 
#endif 



struct qdebug
{
    float wavelength ; 
    float cosTheta ; 
    float3 normal ; 

    qstate s ; 
    quad2  prd ; 
    sphoton  p ; 

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    void save(const char* dir) const ; 
#endif

}; 

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#include "NP.hh"

inline void qdebug::save(const char* dir) const 
{
    s.save(dir);    

    NP* pr = NP::Make<float>(1,2,4) ; 
    pr->read2( (float*)prd.cdata() ); 
    pr->save(dir, "prd.npy");

    NP* ph = NP::Make<float>(1,4,4) ; 
    ph->read2( (float*)p.cdata() ); 
    ph->save(dir, "p0.npy");   
}
#endif


