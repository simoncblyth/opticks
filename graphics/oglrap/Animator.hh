#pragma once
#include "float.h"
#include "string.h"
#include <math.h>
#include <stdio.h>
#include "assert.h"
#include "Interactor.hh"


// TODO: try to support live changing of the range 

class Animator {
    public:
        static const int period_low ; 
        static const int period_high ; 

        static const char* OFF_ ; 
        static const char* SLOW_ ; 
        static const char* NORM_ ; 
        static const char* FAST_ ; 

        typedef enum {  OFF, SLOW, NORM, FAST, NUM_MODE } Mode_t ;


        Animator(float* target, unsigned int period, float low=0.f, float high=1.f);
        void setModeRestrict(Mode_t restrict_);

        void reset();
        bool step(bool& bump); 
        void Summary(const char* msg);

        float* getTarget(); 
        float getLow(); 
        float getHigh(); 
        bool isAnimating();

        bool gui(const char* label, const char* fmt, float power=1.0f);

        unsigned int getNumMode();
        void setMode( Mode_t mode);
        void nextMode(unsigned int modifiers);
        const char* getModeString();

        char* description();

    private:
        void          modeTransition(float fraction);
        void          setTarget(float* target); // qty to be stepped
        unsigned int  getIndex();
        float         getFraction();
        float         getValue();
        float         getFractionForValue(float value);
        unsigned int  find_closest_index(float f);
        bool          isBump();

    private:
        float* make_fractions(unsigned int num);

    private:
        Mode_t       m_mode ; 
        Mode_t       m_restrict ; 
        unsigned int m_period[NUM_MODE] ; 
        float        m_low ; 
        float        m_high ; 
        float*       m_fractions[NUM_MODE] ; 
        unsigned int m_count ; 
        unsigned int m_index ; 
        char         m_desc[32] ; 
        float*       m_target ; 
        int          m_increment ; 
};



inline Animator::Animator(float* target, unsigned int period, float low, float high)
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



inline float Animator::getLow()
{
    return m_low ; 
}
inline float Animator::getHigh()
{
    return m_high ; 
}
inline float* Animator::getTarget()
{
    return m_target ;
}
inline float* Animator::make_fractions(unsigned int num)
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

inline void Animator::setModeRestrict(Mode_t restrict_)
{
    m_restrict = restrict_ ;
}

inline void Animator::setMode(Mode_t mode)
{
    if(mode == m_mode) return ;
    float fraction = getFractionForValue(*m_target);
    m_mode = mode ;  
    modeTransition(fraction);
}


inline unsigned int Animator::getNumMode()
{
    return m_restrict > 0 ? m_restrict : NUM_MODE ;  
}

inline void Animator::nextMode(unsigned int modifiers)
{
    if(modifiers & Interactor::e_shift) m_increment = -m_increment ;

    unsigned int num_mode = getNumMode();
    int next = (m_mode + 1) % num_mode ; 
    setMode((Mode_t)next) ; 
}

inline void Animator::modeTransition(float fraction)
{
    // adjust the count to new raster, to avoid animation jumps 
    if(m_mode == OFF) return ;
    int count = find_closest_index(fraction); 
    printf("Animator::modeTransition fraction %10.3f closest count %d \n", fraction, count ); 
    m_count = count ; 
}


inline bool Animator::isAnimating()
{
    return m_mode != OFF ; 
}


inline void Animator::setTarget(float* target)
{
    m_target = target ;
}


inline bool Animator::isBump()
{
    return m_count > 0 && m_count % m_period[m_mode] == 0 ;
}
inline unsigned int Animator::getIndex()
{
    m_index = m_count % m_period[m_mode] ; // modulo, responsible for the sawtooth
    return m_index ; 
} 

inline float Animator::getFraction()
{
    return m_fractions[m_mode][getIndex()] ;
}
inline float Animator::getValue()
{
    return m_low + (m_high-m_low)*getFraction() ;
}
inline float Animator::getFractionForValue(float value)
{
    return (value - m_low)/(m_high - m_low) ;  
}


 
inline bool Animator::step(bool& bump)
{
    if(m_mode == OFF) return false ; 

    bump = isBump();
    float value = getValue() ;
    
    m_count += m_increment  ;       // NB increment only after getting the value (which depends on m_count) and bump

    *m_target = value ; 

    return true ; 
}
inline void Animator::reset()
{
    m_count = 0 ; 
}


inline unsigned int Animator::find_closest_index(float f )
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




