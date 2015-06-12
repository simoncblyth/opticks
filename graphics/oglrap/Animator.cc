#include "Animator.hh"

const int Animator::period_low  = 25 ; 
const int Animator::period_high = 10000 ; 


void Animator::setPeriod(unsigned int period)
{
    if(period == m_period ) return ; 

    float current_fraction = getFraction();
     
    if(     period < period_low ) m_period = period_low ;
    else if(period > period_high) m_period = period_high ;
    else                          m_period = period ;
 
    m_fractions = make_fractions(m_period);
    m_count = find_closest_index(current_fraction);
 
}

