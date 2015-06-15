#include "Animator.hh"

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal

#ifdef GUI_
#include <imgui.h>
#endif


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



void Animator::gui(const char* label, const char* fmt, float power)
{
#ifdef GUI_

    int prior = m_mode ; 
    float fraction = getFractionForValue(*m_target);

    int* mode = (int*)&m_mode ;   // address of enum cast to int*

    ImGui::SliderFloat( label, m_target, m_low, m_high , fmt, power);
    ImGui::Text("animation mode: ");

    ImGui::RadioButton( OFF_ , mode, OFF); ImGui::SameLine();
    ImGui::RadioButton(SLOW_, mode, SLOW); ImGui::SameLine();
    ImGui::RadioButton(NORM_, mode, NORM); ImGui::SameLine();
    ImGui::RadioButton(FAST_, mode, FAST); //ImGui::SameLine();

    if(m_mode != prior) modeTransition(fraction);
#endif
}


/*
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
*/

