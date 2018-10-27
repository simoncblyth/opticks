
#include <cfloat>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cassert>
#include <sstream>

#include "OpticksConst.hh"
#include "Animator.hh"

#include "PLOG.hh"

const char* Animator::OFF_  = "OFF" ; 
const char* Animator::SLOW32_ = "SLOW32" ; 
const char* Animator::SLOW16_ = "SLOW16" ; 
const char* Animator::SLOW8_ = "SLOW8" ; 
const char* Animator::SLOW4_ = "SLOW4" ; 
const char* Animator::SLOW2_ = "SLOW2" ; 
const char* Animator::NORM_ = "NORM" ; 
const char* Animator::FAST_ = "FAST" ; 
const char* Animator::FAST2_ = "FAST2" ; 
const char* Animator::FAST4_ = "FAST4" ; 

const int Animator::period_low  = 25 ; 
const int Animator::period_high = 10000 ; 


Animator::Animator(float* target, unsigned int period, float low, float high)
    :
    m_mode(OFF),
    m_restrict(NUM_MODE),
    m_low(low),
    m_high(high),
    m_count(0),
    m_index(0),
    m_target(target),
    m_increment(1),
    m_cmd_slots(8),
    m_cmd_index(0),
    m_cmd_offset(0),
    m_cmd_tranche(0)
{
    m_period[OFF]  = 0 ; 
    m_period[SLOW32] = period*32 ; 
    m_period[SLOW16] = period*16  ; 
    m_period[SLOW8] = period*8  ; 
    m_period[SLOW4] = period*4  ; 
    m_period[SLOW2] = period*2  ; 
    m_period[NORM] = period    ; 
    m_period[FAST] = period/2  ; 
    m_period[FAST2] = period/4  ; 
    m_period[FAST4] = period/8  ; 

    m_fractions[OFF]  = NULL ; 
    m_fractions[SLOW32] = make_fractions(m_period[SLOW32]) ;
    m_fractions[SLOW16] = make_fractions(m_period[SLOW16]) ;
    m_fractions[SLOW8] = make_fractions(m_period[SLOW8]) ;
    m_fractions[SLOW4] = make_fractions(m_period[SLOW4]) ;
    m_fractions[SLOW2] = make_fractions(m_period[SLOW2]) ;
    m_fractions[NORM] = make_fractions(m_period[NORM]) ;
    m_fractions[FAST] = make_fractions(m_period[FAST]) ;
    m_fractions[FAST2] = make_fractions(m_period[FAST2]) ;
    m_fractions[FAST4] = make_fractions(m_period[FAST4]) ;

    m_cmd[OFF]  = "T0" ; 
    m_cmd[SLOW32] = "T1" ; 
    m_cmd[SLOW16] = "T2" ; 
    m_cmd[SLOW8] = "T3" ; 
    m_cmd[SLOW4] = "T4" ; 
    m_cmd[SLOW2] = "T5" ; 
    m_cmd[NORM] = "T6" ; 
    m_cmd[FAST] = "T7" ; 
    m_cmd[FAST2] = "T8" ; 
    m_cmd[FAST4] = "T9" ; 

}

/**
    SLOW32 128*32      4096
           128*16      2048
    SLOW8  128*8       1024
           128*4        512
    SLOW2  128*2        256
    NORM   128          128
    FAST   128/2         64
    FAST2  128/4         32
    FAST4  128/8         16
**/


Animator::Mode_t Animator::getMode()
{
    return m_mode ; 
}

bool Animator::isModeChanged(Mode_t prior)
{
    return m_mode != prior ; 
}


int* Animator::getModePtr()
{
    int* mode = (int*)&m_mode ;   // address of enum cast to int*
    return mode ; 
}


bool Animator::isSlowEnabled()
{
    return SLOW2 < m_restrict ; 
}
bool Animator::isNormEnabled()
{
    return NORM < m_restrict ; 
}
bool Animator::isFastEnabled()
{
    return FAST < m_restrict ; 
}







float Animator::getLow()
{
    return m_low ; 
}
float Animator::getHigh()
{
    return m_high ; 
}
float* Animator::getTarget()
{
    return m_target ;
}
float* Animator::make_fractions(unsigned int num)
{
    float low(0.f);
    float high(1.f);

    float* frac = new float[num] ;
    float step_ = (high - low)/float(num-1)  ;
    // from i=0 to i=num-1 
    for(unsigned int i=0 ; i < num ; i++) frac[i] = low + step_*i ; 
    return frac ; 

// In [13]: np.linspace(0.,1.,11)
// Out[13]: array([ 0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1. ])
}

void Animator::setModeRestrict(Mode_t restrict_)
{
    m_restrict = restrict_ ;
}

void Animator::setMode(Mode_t mode)
{
    if(mode == m_mode) return ;
    float fraction = getFractionForValue(*m_target);
    m_mode = mode ;  

    LOG(info) << description() ; 

    modeTransition(fraction);
}


unsigned int Animator::getNumMode()
{
    return m_restrict > 0 ? m_restrict : NUM_MODE ;  
}


void Animator::commandMode(const char* cmd)
{
    //LOG(info) << cmd ; 

    assert(strlen(cmd) == 2); 
    assert( cmd[0] == 'T' || cmd[0] == 'A' ); 
      
    int mode = (int)cmd[1] - (int)'0' ; 
    assert( mode > -1 && mode < NUM_MODE ) ; 

    setMode((Mode_t)mode) ; 
}

void Animator::nextMode(unsigned int modifiers)
{
    if(modifiers & OpticksConst::e_shift) m_increment = -m_increment ;

    bool option = 0 != (modifiers & OpticksConst::e_option) ;    
    bool control = 0 != (modifiers & OpticksConst::e_control) ;    

    unsigned int num_mode = getNumMode();

    int mode = ( option ? m_mode - 1 : m_mode + 1) % num_mode ; 

    if(mode < 0) mode = num_mode - 1 ;  

    if(control) mode = OFF ; 

    setMode((Mode_t)mode) ; 
}

void Animator::modeTransition(float fraction)
{
    // adjust the count to new raster, to avoid animation jumps 
    if(m_mode == OFF) return ;
    int count = find_closest_index(fraction); 

#ifdef ANIMATOR_DEBUG
    //printf("Animator::modeTransition fraction %10.3f closest count %d \n", fraction, count ); 
#endif

    m_count = count ; 
}


bool Animator::isActive()
{
    return m_mode != OFF ; 
}


void Animator::setTarget(float* target)
{
    m_target = target ;
}

void Animator::scrub_to(float , float , float , float dy) // Interactor:K scrub_mode
{
   // hmm maybe easier to make separate mostly transparent ImGui window with just the time scrubber
   // to avoid wheel reinvention
    if(m_mode == OFF) return ; 

    float val = getValue();
    val += 30.f*dy ; 


    LOG(info) << "Animator::scrub_to"
              << " dy " << dy
              << " val " << val 
              ;

    setTargetValue(val);
}

void Animator::setTargetValue(float val)
{
    if(val < m_low)         val = m_high ;
    else if( val > m_high ) val = m_low  ; 

    float f = getFractionForValue(val);
    setFraction(f);
}

void Animator::setFraction(float f)
{
    *m_target = m_low + (m_high-m_low)*f ; 
}


bool Animator::isBump()
{
    return m_count > 0 && m_count % m_period[m_mode] == 0 ;
}
unsigned int Animator::getIndex()
{
    m_index = m_count % m_period[m_mode] ; // modulo, responsible for the sawtooth
    return m_index ; 
} 

float Animator::getFraction()
{
    unsigned int index = getIndex();
    return m_fractions[m_mode][index] ;
}
float Animator::getValue()
{
    return m_low + (m_high-m_low)*getFraction() ;
}
float Animator::getFractionForValue(float value)
{
    return (value - m_low)/(m_high - m_low) ;  
}
 
float Animator::getFractionFromTarget()
{
    return getFractionForValue(*m_target);
}

bool Animator::step(bool& bump, unsigned& cmd_index, unsigned& cmd_offset )
{
    bool st = step(bump) ; 
    if(!st) return st ; 

    m_cmd_tranche = m_period[m_mode]/m_cmd_slots ; //  NB keep animator_period a suitable power of two, such as 128
    m_cmd_index = m_index/m_cmd_tranche ;  
    assert( m_cmd_index < m_cmd_slots ) ;    
    m_cmd_offset = m_index - m_cmd_index*m_cmd_tranche ;
    assert( m_cmd_offset < m_cmd_tranche ) ;

    cmd_index = m_cmd_index ; 
    cmd_offset = m_cmd_offset ;     


    return st ; 
}
 

bool Animator::step(bool& bump)
{
   // still seeing occasional jumps, but cannot reproduce

    if(m_mode == OFF) return false ; 

    bump = isBump();

#ifdef ANIMATOR_DEBUG
    if(bump)
       printf("Animator::step bump m_count %d \n", m_count );
#endif


    float value = getValue() ;
    
    m_count += m_increment  ;       // NB increment only after getting the value (which depends on m_count) and bump

    *m_target = value ; 

    return true ; 
}


void Animator::reset()
{
    m_count = 0 ; 
}


void Animator::home()
{
    m_count = 0 ; 
    bool bump(false);
    step(bump);
}



unsigned int Animator::find_closest_index(float f )
{
    float fmin(FLT_MAX);
    int ic(-1);

    unsigned int period = m_period[m_mode];
    //printf("Animator::find_closest_index f %10.3f period %d \n", f, period);
    for(unsigned int i=0 ; i < period ; i++)  
    {
        float ifrac = m_fractions[m_mode][i];
        float diff = fabs(f - ifrac) ;

        //printf(" i %d ifrac %10.4f diff %10.4f ic %d \n", i, ifrac, diff, ic ) ;
        if( diff < fmin )
        {
            fmin = diff ;
            ic = i ;
        }
    }
    return ic ; 
}












char* Animator::description()
{
    snprintf(m_desc, 64, " %2s:%5s %d/%d/%10.4f", getModeCmd(), getModeName() , m_index, m_period[m_mode], *m_target );
    return m_desc ; 
}

void Animator::Summary(const char* msg)
{
    LOG(info) << msg << description() ; 
}


const char* Animator::getModeCmd() const
{
    return m_cmd[m_mode] ; 
}
 
const char* Animator::getModeName() const 
{
    const char* mode(NULL);
    switch(m_mode)
    {
        case  OFF:mode = OFF_ ; break ; 
        case SLOW32:mode = SLOW32_ ; break ; 
        case SLOW16:mode = SLOW16_ ; break ; 
        case SLOW8:mode = SLOW8_ ; break ; 
        case SLOW4:mode = SLOW4_ ; break ; 
        case SLOW2:mode = SLOW2_ ; break ; 
        case NORM:mode = NORM_ ; break ; 
        case FAST:mode = FAST_ ; break ; 
        case FAST2:mode = FAST2_ ; break ; 
        case FAST4:mode = FAST4_ ; break ; 
        case NUM_MODE:assert(0) ; break ; 
    }
    return mode ; 
}

std::string Animator::desc() const 
{
    std::stringstream ss ; 
    ss  << "Animator "
        << getModeName()
        << " ci:" << m_cmd_index
        << " co:" << m_cmd_offset
        << " ct:" << m_cmd_tranche 
        ;
    return ss.str(); 
}



