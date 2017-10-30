#pragma once

//#include <boost/random/mersenne_twister.hpp>
#include <boost/random/inversive_congruential.hpp>

#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>


#include "BRAP_API_EXPORT.hh"

class BRAP_API BRng 
{
    //typedef boost::mt19937          RNG_t;
    typedef boost::hellekalek1995   RNG_t ; 
    typedef boost::uniform_real<>   Distrib_t;
    typedef boost::variate_generator< RNG_t, Distrib_t > Generator_t ;

    public:
        BRng(float lo=0.f, float hi=1.f);
        float operator()();
    private:
        RNG_t       m_rng;
        Distrib_t   m_dist ;
        Generator_t m_gen ; 
};

