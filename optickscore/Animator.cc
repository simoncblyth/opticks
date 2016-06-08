#include "Animator.hh"

#include "BLog.hh"


const char* Animator::OFF_  = "OFF" ; 
const char* Animator::SLOW_ = "SLOW" ; 
const char* Animator::NORM_ = "NORM" ; 
const char* Animator::FAST_ = "FAST" ; 

const int Animator::period_low  = 25 ; 
const int Animator::period_high = 10000 ; 


char* Animator::description()
{
    snprintf(m_desc, 64, " %5s %d/%d/%10.4f", getModeString() , m_index, m_period[m_mode], *m_target );
    return m_desc ; 
}

void Animator::Summary(const char* msg)
{
    LOG(info) << msg << description() ; 
}

const char* Animator::getModeString()
{
    const char* mode(NULL);
    switch(m_mode)
    {
        case  OFF:mode = OFF_ ; break ; 
        case SLOW:mode = SLOW_ ; break ; 
        case NORM:mode = NORM_ ; break ; 
        case FAST:mode = FAST_ ; break ; 
        case NUM_MODE:assert(0) ; break ; 
    }
    return mode ; 
}





