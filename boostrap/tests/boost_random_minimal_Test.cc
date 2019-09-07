/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */


#include <iostream>
#include <fstream>
#include <ctime>            // std::time

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/linear_congruential.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/generator_iterator.hpp>

#include "PLOG.hh"

//typedef boost::mt19937     RNG_t; 
//typedef boost::ecuyer1988  RNG_t;
typedef boost::minstd_rand      RNG_t;

typedef boost::uniform_real<>   DST_t;
typedef boost::variate_generator< RNG_t, DST_t > GEN_t ;



class BRNG
{
    public:
        BRNG( float lo=0.f, float hi=1.f, unsigned _seed=0)
            :
            m_rng(),
            m_dst(lo, hi),
            m_gen(m_rng, m_dst)
        {
            seed(_seed); 
        }

        float operator()()
        {
            return m_gen() ;   
        }

        void seed(unsigned _seed)
        {
            m_rng.seed(_seed);
        }

        void dump( const char* label )
        {
            LOG(info) << label ; 
            std::cout.setf(std::ios::fixed);
            for(int i = 0; i < 10; i++) std::cout << (*this)() << '\n';
        }


    private:
        RNG_t m_rng ;
        DST_t m_dst ;
        GEN_t m_gen ; 
};


void test_0()
{
    LOG(info) << "." ;

    { 
        RNG_t rng(42);
        DST_t dst(0,1);
        GEN_t uni(rng, dst);

        std::cout << "A : 10 samples in [0..1) (seed:42) \n";
        std::cout.setf(std::ios::fixed);
        for(int i = 0; i < 10; i++) std::cout << uni() << '\n';

        rng.seed(420);
        std::cout << "A : 10 samples in [0..1) (seed:420) \n";
        for(int i = 0; i < 10; i++) std::cout << uni() << '\n';
    } 

    { 

        RNG_t rng(42);
        DST_t dst(0,1);
        GEN_t uni(rng, dst);

        std::cout << "B : 10 samples of a uniform distribution in [0..1):\n";
        std::cout.setf(std::ios::fixed);
        for(int i = 0; i < 10; i++) std::cout << uni() << '\n';

        rng.seed(420);
        std::cout << "B : 10 more samples of a uniform distribution in [0..1):\n";
        for(int i = 0; i < 10; i++) std::cout << uni() << '\n'; 
    } 

}

void test_1()
{
    LOG(info) << "." ;
    {
        BRNG a(0,1,42);
        a.dump("A : 10 samp, seed:42 ");
        a.seed(420);
        a.dump("A : 10 samp, seed 420 ");
    }
    {
        BRNG b(0,1,42);
        b.dump("B : 10 samp, seed:42 ");
        b.seed(420);
        b.dump("B : 10 samp, seed 420 ");
    }




}





int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    test_0();
    test_1();

    return 0;
}
