#pragma once

#include <string>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/inversive_congruential.hpp>

#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>

/**

BRng
======

* http://www.boost.org/doc/libs/1_65_1/doc/html/boost_random/tutorial.html

* https://stackoverflow.com/questions/4778797/setting-seed-boostrandom


**/


#include "BRAP_API_EXPORT.hh"

class BRAP_API BRng 
{
    typedef boost::mt19937          RNG_t;
    //typedef boost::hellekalek1995   RNG_t ; 
    typedef boost::uniform_real<>   DST_t;
    typedef boost::variate_generator< RNG_t, DST_t > GEN_t ;

    public:
        BRng(float lo=0.f, float hi=1.f, unsigned seed=42, const char* label="A");
        float one();
        void two(float& a, float& b);
        void setSeed(unsigned _seed);
        std::string desc() const ;
        void dump(); 
    private:
        float m_lo ; 
        float m_hi ; 

        RNG_t* m_rng;
        DST_t* m_dst ;
        GEN_t* m_gen ; 

        unsigned m_seed ; 
        const char* m_label ; 
        unsigned m_count ; 
        
   
};

