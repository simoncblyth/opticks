#include "BRng.hh"

BRng::BRng(float low, float high) 
   :   
   m_dist(low, high),
   m_gen(m_rng, m_dist)
{
}

float BRng::operator()()
{
    return m_gen() ;
}


