#pragma once

//#include <boost/random/mersenne_twister.hpp>
#include <boost/random/inversive_congruential.hpp>

#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>

#include "NBBox.hpp"

#include "NPY_API_EXPORT.hh"

class NPY_API NGenerator 
{
    //typedef boost::mt19937          RNG_t;
    typedef boost::hellekalek1995   RNG_t ; 
    typedef boost::uniform_real<>   Distrib_t;
    typedef boost::variate_generator< RNG_t, Distrib_t > Generator_t ;

    public:
        NGenerator(const nbbox& bb);
        void operator()(nvec3& xyz);
    private:
        nbbox m_bb ; 
        nvec3 m_side ; 
        RNG_t m_rng;
        Distrib_t m_dist ;
        Generator_t m_gen ; 
};



