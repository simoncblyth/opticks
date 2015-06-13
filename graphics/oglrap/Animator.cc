#include "Animator.hh"

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal




const int Animator::period_low  = 25 ; 
const int Animator::period_high = 10000 ; 


void Animator::setPeriod(unsigned int period)
{
    if(period == m_period ) return ; 

    float current_fraction = getFraction();
     
    if(     period < period_low ) m_period = period_low ;
    else if(period > period_high) m_period = period_high ;
    else                          m_period = period ;
 
    delete m_fractions ; 
    m_fractions = make_fractions(m_period);
    m_count = find_closest_index(current_fraction);
}

char* Animator::description()
{
    snprintf(m_desc, 64, " %3s %d/%d/%10.4f", m_on ? "ON" : "OFF" , m_index, m_period, *m_target );
    return m_desc ; 
}


void Animator::Summary(const char* msg)
{
    LOG(info) << msg << description() ; 
}

