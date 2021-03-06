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

#pragma once

#include <string>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/inversive_congruential.hpp>

#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>

#include "plog/Severity.h"

/**

BRng
======

* http://www.boost.org/doc/libs/1_65_1/doc/html/boost_random/tutorial.html

* https://stackoverflow.com/questions/4778797/setting-seed-boostrandom


**/


#include "BRAP_API_EXPORT.hh"

class BRAP_API BRng 
{
    static const plog::Severity LEVEL ;  

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

        float getLo() const ; 
        float getHi() const ; 
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

