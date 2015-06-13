#pragma once
#include "float.h"
#include "string.h"
#include <math.h>
#include <stdio.h>

class Animator {
    public:
        static const int period_low ; 
        static const int period_high ; 

        Animator(float* target, unsigned int period, float low=0.f, float high=1.f);

        void reset();
        float step(bool& bump); 
        void Summary(const char* msg);

        float getLow(); 
        float getHigh(); 
        float* getTarget(); 

       
        void setPeriod(unsigned int period);
        char* description();
        void scalePeriod(float factor);

        void setOn(bool on=true);
        bool isOn();
        bool* isOnPtr();
        void toggle();

    private:
        void setTarget(float* target); // qty to be stepped
        unsigned int  getIndex();
        float         getFraction();
        unsigned int  find_closest_index(float f);
        bool          isBump();




    private:
        float* make_fractions(unsigned int num, float low=0.f , float high=1.f);

    private:
        bool         m_on ; 
        unsigned int m_period ; 
        float        m_low ; 
        float        m_high ; 
        float*       m_fractions ; 
        unsigned int m_count ; 
        unsigned int m_index ; 
        char         m_desc[32] ; 
        float*       m_target ; 

};



inline Animator::Animator(float* target, unsigned int period, float low, float high)
    :
    m_on(false),
    m_period(period),
    m_low(low),
    m_high(high),
    m_fractions(make_fractions(period,0.f,1.f)),
    m_count(0),
    m_index(0),
    m_target(target)
{
}

inline void Animator::setTarget(float* target)
{
    m_target = target ;
}


inline bool Animator::isOn()
{
   return m_on ; 
}
inline bool* Animator::isOnPtr()
{
   return &m_on ; 
}

inline void Animator::setOn(bool on)
{
   m_on = on ;  
}
inline void Animator::toggle()
{
   m_on = !m_on ;  
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
    float value = m_low + (m_high-m_low)*fraction ;

    if(m_on)
    {
        // NB increments only when active and after getting the fraction and bump
        m_count += 1 ; 
        if(m_target) *m_target = value ; 
    } 

    //printf("Animator::step m_on %d m_count %d value %10.4f \n", m_on, m_count,  value );      
    return value ; 
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



