
// either just umbrella header OR 4-specific ones works
//#include "boost/random.hpp"

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>

#include "stdio.h"


void dice_demo()
{
    // http://stackoverflow.com/questions/2254909/boost-random-number-generator
    typedef boost::mt19937       RNG_t;
    typedef boost::uniform_int<> Distrib_t;
    typedef boost::variate_generator< RNG_t, Distrib_t > Generator_t ;

    RNG_t rng;
    Distrib_t distrib(1,6);
    Generator_t gen(rng, distrib);

    for(int i = 0; i < 10; i++ ) printf("%d\n", gen()) ;
}


void real_demo()
{
    typedef boost::mt19937          RNG_t;
    typedef boost::uniform_real<>   Distrib_t;
    typedef boost::variate_generator< RNG_t, Distrib_t > Generator_t ;

    RNG_t rng;
    Distrib_t distrib(0,1);
    Generator_t gen(rng, distrib);

    for(int i = 0; i < 10; i++ ) printf("%15.8f\n", gen()) ;
}


int main() 
{
    dice_demo();
    real_demo();
    return 0 ;
}
