#pragma once

#include <random>

struct srng
{
    std::mt19937_64 engine ;

    std::uniform_real_distribution<float>   fdist ; 
    std::uniform_real_distribution<double>  ddist ; 
    double fake ; 

    srng(unsigned seed_); 

    void set_fake(double fake_); 
    float  generate_float(); 
    double generate_double(); 


    static float  uniform(srng* state );
    static double uniform_double(srng* state ); 

}; 


inline srng::srng(unsigned seed_) 
    : 
    fdist(0,1), 
    ddist(0,1),
    fake(-1.)
{ 
    engine.seed(seed_) ; 
}

inline void   srng::set_fake(double fake_){ fake = fake_ ; } 
inline float  srng::generate_float(){  return fake >= 0.f ? fake : fdist(engine) ; } 
inline double srng::generate_double(){ return fake >= 0. ?  fake : ddist(engine) ; } 


inline float  srng::uniform(srng* state ){        return state->generate_float() ; } 
inline double srng::uniform_double(srng* state ){ return state->generate_double() ; } 




