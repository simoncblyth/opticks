#pragma once
/**
qdebug.h
==========

Instanciation managed from QSim

**/

#include "scuda.h"
#include "squad.h"
#include "sphoton.h"
#include "sscint.h"

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
    quad6    scint_gs ; 

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    void save(const char* dir) const ; 
    void save_scint_gs(const char* dir) const ; 
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

inline void qdebug::save_scint_gs(const char* dir) const 
{
    NP* gs = NP::Make<float>(1,6,4) ; 
    gs->read2( (float*)scint_gs.cdata() ); 
    gs->save(dir, "scint_gs.npy"); 
}


#endif


