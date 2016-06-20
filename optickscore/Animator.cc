
#include <cfloat>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cassert>


#include "BLog.hh"
#include "OpticksConst.hh"
#include "Animator.hh"


const char* Animator::OFF_  = "OFF" ; 
const char* Animator::SLOW_ = "SLOW" ; 
const char* Animator::NORM_ = "NORM" ; 
const char* Animator::FAST_ = "FAST" ; 

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
    m_increment(1)
{
    m_period[OFF]  = 0 ; 
    m_period[SLOW] = period*2  ; 
    m_period[NORM] = period    ; 
    m_period[FAST] = period/2  ; 

    m_fractions[OFF]  = NULL ; 
    m_fractions[SLOW] = make_fractions(m_period[SLOW]) ;
    m_fractions[NORM] = make_fractions(m_period[NORM]) ;
    m_fractions[FAST] = make_fractions(m_period[FAST]) ;
}


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
    return SLOW < m_restrict ; 
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
    float step = (high - low)/float(num-1)  ;
    // from i=0 to i=num-1 
    for(unsigned int i=0 ; i < num ; i++) frac[i] = low + step*i ; 
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
    modeTransition(fraction);
}


unsigned int Animator::getNumMode()
{
    return m_restrict > 0 ? m_restrict : NUM_MODE ;  
}

void Animator::nextMode(unsigned int modifiers)
{
    if(modifiers & OpticksConst::e_shift) m_increment = -m_increment ;

    bool option = modifiers & OpticksConst::e_option ;    
    bool control = modifiers & OpticksConst::e_control ;    

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

void Animator::scrub_to(float x, float y, float dx, float dy) // Interactor:K scrub_mode
{
   // hmm maybe easier to make separate mostly transparent ImGui window with just the time scrubber
   // to avoid wheel reinvention
    if(m_mode == OFF) return ; 

    float val = getValue();
    val += 30.*dy ; 
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





