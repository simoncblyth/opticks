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

#include "PLOG.hh"
#include "BRng.hh"


const plog::Severity BRng::LEVEL = PLOG::EnvLevel("BRng", "DEBUG") ; 


float BRng::getLo() const 
{
    return m_lo ;
}
float BRng::getHi() const 
{
    return m_hi ;
}


BRng::BRng(float lo, float hi, unsigned _seed, const char* label) 
   :   
   m_lo(lo),
   m_hi(hi),
   m_rng(NULL),   //  cannot give a ctor arg to m_rng here ?
   m_dst(NULL),
   m_gen(NULL),
   m_label( label ? strdup(label) : "?" ),
   m_count(0)
{
    setSeed(_seed);
}


float BRng::one()
{   
    m_count++ ; 
    return (*m_gen)() ;
}


void BRng::two(float& a, float& b)
{   
    a = one(); 
    b = one();
}


void BRng::setSeed(unsigned _seed)
{
    LOG(LEVEL) << m_label << " setSeed(" << _seed << ")" ; 

    m_seed = _seed ; 

    // forced to recreate as trying to seed/reset 
    // existing ones failed to giving a fresh sequence

    delete m_gen ; 
    delete m_dst ; 
    delete m_rng ; 

    m_rng = new RNG_t(m_seed) ; 
    m_dst = new DST_t(m_lo, m_hi);
    m_gen = new GEN_t(*m_rng, *m_dst) ;

}

std::string BRng::desc() const 
{
    std::stringstream ss ; 

    ss << m_label 
       << " "
       << " seed " << m_seed
       << " lo " << m_lo
       << " hi " << m_hi
       << " count " << m_count
       ;

    return ss.str();
}

void BRng::dump()
{
    LOG(info) << desc() ; 
    for(unsigned i=0 ; i < 10 ; i++ ) 
       std::cout << one() << std::endl ; 
}






