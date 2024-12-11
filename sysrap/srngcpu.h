#pragma once
/**
srngcpu.h : C++ standard random number generation
=================================================

Instead of generating randoms it is also possible to 
use curand precooked randoms by calling the below method 
with the photon index as argument::

    srngcpu::setSequenceIndex

This is done by SGenerate::GeneratePhotons when the below EKEY is set::

    export SGenerate__GeneratePhotons_RNG_PRECOOKED=1

Using this limits the number of photons that can be
generated to the number of rng_sequence that have been precooked 
and persisted to ~/.opticks/precooked.
To extend that see::
  
   ~/opticks/qudarap/tests/rng_sequence.sh

**/

#include <random>
#include "s_seq.h"

struct srngcpu
{
    int                                     seed ;  
    std::mt19937_64                         engine ;
    std::uniform_real_distribution<float>   fdist ; 
    std::uniform_real_distribution<double>  ddist ; 
    double                                  fake ; 
    s_seq*                                  seq ; 


    srngcpu(); 
    std::string desc() const ; 

    void set_fake(double fake_); 
    void setSequenceIndex(int idx); 
    int  getSequenceIndex() const ; 

    float  generate_float(); 
    double generate_double(); 

    static float  uniform(srngcpu* state );
    static double uniform_double(srngcpu* state ); 

    std::string demo(int n) ; 
}; 


inline srngcpu::srngcpu() 
    : 
    seed(1),
    fdist(0,1), 
    ddist(0,1),
    fake(-1.),
    seq(nullptr)
{ 
    engine.seed(seed) ; 
}

inline std::string srngcpu::desc() const 
{
    std::stringstream ss ; 
    ss << "srngcpu::desc" << std::endl ; 
    std::string str = ss.str(); 
    return str ; 
}

inline void srngcpu::set_fake(double fake_){ fake = fake_ ; } 

inline float srngcpu::generate_float()
{
    if( fake >= 0.f ) return fake ; 
    float u = seq && seq->is_enabled() ? seq->flat() : fdist(engine) ; 
    return u ; 
} 
inline double srngcpu::generate_double()
{ 
    if( fake >= 0.f ) return fake ; 
    double u = seq && seq->is_enabled() ? seq->flat() : ddist(engine) ; 
    return u ; 
}
inline void srngcpu::setSequenceIndex(int idx)
{
    if( seq == nullptr ) seq = new s_seq ; 
    seq->setSequenceIndex(idx); 
}
inline int srngcpu::getSequenceIndex() const 
{
    return seq == nullptr ? -2 : seq->getSequenceIndex() ; 
}



inline float  srngcpu::uniform(srngcpu* state ){        return state->generate_float() ; } 
inline double srngcpu::uniform_double(srngcpu* state ){ return state->generate_double() ; } 

inline std::string srngcpu::demo(int n) 
{
    std::stringstream ss ; 
    ss << "srngcpu::demo seq " << getSequenceIndex()  << std::endl ;
    for(int i=0 ; i < n ; i++) ss 
         << std::setw(4) << i 
         << " : " 
         << std::fixed << std::setw(10) << std::setprecision(5) << generate_float() 
         << std::endl
         ; 
    std::string str = ss.str(); 
    return str ; 
}

// "mocking" the curand API 
inline float  curand_uniform(srngcpu* state ){         return state->generate_float() ; }
inline double curand_uniform_double(srngcpu* state ){ return state->generate_double() ; }





