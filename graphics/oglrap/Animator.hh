#pragma once
#include "float.h"
#include "string.h"
#include <math.h>
#include <stdio.h>

class Animator {
    public:
        static const int period_low ; 
        static const int period_high ; 

        Animator(unsigned int period, float low=0.f, float high=1.f);
        void reset();
        float step(bool& bump); 
        void setPeriod(unsigned int period);
        char* description();
        void scalePeriod(float factor);

    private:
        unsigned int  getIndex();
        float         getFraction();
        unsigned int  find_closest_index(float f);
        bool          isBump();



    private:
        float* make_fractions(unsigned int num, float low=0.f , float high=1.f);

    private:
        unsigned int m_period ; 
        float        m_low ; 
        float        m_high ; 
        float*       m_fractions ; 
        unsigned int m_count ; 
        unsigned int m_index ; 
        char         m_desc[32] ; 

};



inline Animator::Animator(unsigned int period, float low, float high)
    :
    m_period(period),
    m_low(low),
    m_high(high),
    m_fractions(make_fractions(period)),
    m_count(0),
    m_index(0)
{
}

inline void Animator::scalePeriod(float factor)
{
    setPeriod(m_period*factor);
}

inline bool Animator::isBump()
{
    return m_count > 0 && m_count % m_period == 0 ;
}

inline float Animator::step(bool& bump)
{
    bump = isBump();
    float fraction = getFraction();
    // NB increments only after getting the fraction and bump
    m_count += 1 ; 
    return m_low + (m_high-m_low)*fraction ;
}


inline void Animator::reset()
{
    m_count = 0 ; 
}

inline unsigned int Animator::getIndex()
{
    m_index = m_count % m_period ; // modulo, responsible for the sawtooth
    return m_index ; 
} 

inline float Animator::getFraction()
{
    return m_fractions[getIndex()] ;
} 

inline unsigned int Animator::find_closest_index(float f )
{
    float c(FLT_MAX);
    int ic(-1);
    for(unsigned int i=0 ; i < m_period ; i++)  
    {
        float diff = fabs(f - m_fractions[i]) ;
        if( diff < c )
        {
            c = m_fractions[i];
            ic = i ;
        }
    }
    return ic ; 
}



inline float* Animator::make_fractions(unsigned int num, float low, float high)
{
    float* frac = new float[num] ;
    float step = (high - low)/float(num-1)  ;
    // from i=0 to i=num-1 
    for(unsigned int i=0 ; i < num ; i++) frac[i] = low + step*i ; 
    return frac ; 

// In [13]: np.linspace(0.,1.,11)
// Out[13]: array([ 0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1. ])

}


inline char* Animator::description()
{
    snprintf(m_desc, 32, "%d/%d", m_index, m_period );
    return m_desc ; 
}



