#pragma once

#include <random>
#include "s_seq.h"

struct srng
{
    std::mt19937_64                         engine ;
    std::uniform_real_distribution<float>   fdist ; 
    std::uniform_real_distribution<double>  ddist ; 
    double                                  fake ; 
    s_seq*                                  seq ; 

    srng(unsigned seed_=1); 
    std::string desc() const ; 

    void set_fake(double fake_); 
    void setSequenceIndex(int idx); 
    int  getSequenceIndex() const ; 

    float  generate_float(); 
    double generate_double(); 

    static float  uniform(srng* state );
    static double uniform_double(srng* state ); 

    std::string demo(int n) ; 
}; 


inline srng::srng(unsigned seed_) 
    : 
    fdist(0,1), 
    ddist(0,1),
    fake(-1.),
    seq(nullptr)
{ 
    engine.seed(seed_) ; 
}

inline std::string srng::desc() const 
{
    std::stringstream ss ; 
    ss << "srng::desc" << std::endl ; 
    std::string str = ss.str(); 
    return str ; 
}

inline void srng::set_fake(double fake_){ fake = fake_ ; } 

inline float srng::generate_float()
{
    if( fake >= 0.f ) return fake ; 
    return seq && seq->is_enabled() ? seq->flat() : fdist(engine) ; 
} 
inline double srng::generate_double()
{ 
    if( fake >= 0.f ) return fake ; 
    return seq && seq->is_enabled() ? seq->flat() : ddist(engine) ; 
}
inline void srng::setSequenceIndex(int idx)
{
    if( seq == nullptr ) seq = new s_seq ; 
    seq->setSequenceIndex(idx); 
}
inline int srng::getSequenceIndex() const 
{
    return seq == nullptr ? -2 : seq->getSequenceIndex() ; 
}



inline float  srng::uniform(srng* state ){        return state->generate_float() ; } 
inline double srng::uniform_double(srng* state ){ return state->generate_double() ; } 

inline std::string srng::demo(int n) 
{
    std::stringstream ss ; 
    ss << "srng::demo seq " << getSequenceIndex()  << std::endl ;
    for(int i=0 ; i < n ; i++) ss 
         << std::setw(4) << i 
         << " : " 
         << std::fixed << std::setw(10) << std::setprecision(5) << generate_float() 
         << std::endl
         ; 
    std::string str = ss.str(); 
    return str ; 
}



