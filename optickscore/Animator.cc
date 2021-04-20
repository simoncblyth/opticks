/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */


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
const char* Animator::FAST8_ = "FAST8" ; 
const char* Animator::FAST16_ = "FAST16" ; 
const char* Animator::FAST32_ = "FAST32" ; 
const char* Animator::FAST64_ = "FAST64" ; 

const int Animator::LEVEL_MIN = -5 ; 
const int Animator::LEVEL_MAX =  7 ; 


/**
Animator::setPeriod
---------------------

Adjusts speed mode to achieve the target_period  

**/
void Animator::setModeForPeriod(unsigned target_period) 
{
    int target_level = -100 ; 
    for(int level = LEVEL_MIN ; level <= LEVEL_MAX ; level++)
    {
        unsigned period = level == 0 ? m_base_period : ( level < 0 ? m_base_period << -level : m_base_period >> level ) ; 
        if(period == target_period) target_level = level ; 
        LOG(LEVEL) 
            << " level "         << std::setw(3) << level 
            << " period "        << std::setw(6) << period 
            << " target_period " << std::setw(6) << target_period 
            << " target_level "  << std::setw(3) << target_level 
            ; 
    } 
    assert( target_level >= LEVEL_MIN && target_level <= LEVEL_MAX ); 

    LOG(LEVEL)
        << " base_period " << m_base_period
        << " target_period " << target_period
        << " target_level " << target_level
        << " LEVEL_MIN " << LEVEL_MIN 
        << " LEVEL_MAX " << LEVEL_MAX 
        ;

    setMode(target_level); 
}





Animator::Mode_t Animator::Mode(int level)  // static 
{
    Mode_t mode = OFF ; 
    switch(level)
    {
        case -5: mode = SLOW32  ; break ; 
        case -4: mode = SLOW16  ; break ; 
        case -3: mode = SLOW8   ; break ; 
        case -2: mode = SLOW4   ; break ; 
        case -1: mode = SLOW2   ; break ; 
        case  0: mode = NORM    ; break ; 
        case  1: mode = FAST    ; break ; 
        case  2: mode = FAST2   ; break ; 
        case  3: mode = FAST4   ; break ; 
        case  4: mode = FAST8   ; break ; 
        case  5: mode = FAST16  ; break ; 
        case  6: mode = FAST32  ; break ; 
        case  7: mode = FAST64  ; break ; 
        case  8: mode = OFF     ; break ;  
    } 
    return mode ; 
}

Animator::Mode_t Animator::Mode(const char* name)  // static 
{
    Mode_t mode = OFF ; 
    if(     strcmp(name, OFF_) == 0)    mode = OFF ; 
    else if(strcmp(name, SLOW32_) == 0) mode = SLOW32 ; 
    else if(strcmp(name, SLOW16_) == 0) mode = SLOW16 ; 
    else if(strcmp(name, SLOW8_) == 0)  mode = SLOW8 ; 
    else if(strcmp(name, SLOW4_) == 0)  mode = SLOW4 ; 
    else if(strcmp(name, SLOW2_) == 0)  mode = SLOW2 ; 
    else if(strcmp(name, NORM_)  == 0)  mode = NORM ; 
    else if(strcmp(name, FAST2_) == 0)  mode = FAST2 ; 
    else if(strcmp(name, FAST4_) == 0)  mode = FAST4 ; 
    else if(strcmp(name, FAST8_) == 0)  mode = FAST8 ; 
    else if(strcmp(name, FAST16_) == 0) mode = FAST16 ; 
    else if(strcmp(name, FAST32_) == 0) mode = FAST32 ; 
    else if(strcmp(name, FAST64_) == 0) mode = FAST64 ; 
    return mode ;  
}

void Animator::setMode(int level)
{
    setMode(Mode(level));  
}
void Animator::setMode(const char* name)
{
    setMode(Mode(name));  
}





const int Animator::period_low  = 25 ; 
const int Animator::period_high = 10000 ; 


const plog::Severity Animator::LEVEL = PLOG::EnvLevel("Animator", "DEBUG"); 


Animator::Animator(float* target, unsigned int period, float low, float high, const char* label)
    :
    m_mode(OFF),
    m_restrict(NUM_MODE),
    m_base_period(period),
    m_low(low),
    m_high(high),
    m_label(strdup(label)),
    m_count(0),
    m_index(0),
    m_target(target),
    m_increment(1),
    m_cmd_slots(8),
    m_cmd_index(0),
    m_cmd_offset(0),
    m_cmd_tranche(0)
{
    LOG(LEVEL) 
        << " label " << label 
        << " period " << period 
        << " low " << low 
        << " high " << high 
        ;

    m_period[OFF]  = 0 ; 
    m_period[SLOW32] = period << 5 ; 
    m_period[SLOW16] = period << 4 ; 
    m_period[SLOW8] = period << 3 ; 
    m_period[SLOW4] = period << 2 ; 
    m_period[SLOW2] = period << 1 ; 
    m_period[NORM] = period    ; 
    m_period[FAST] = period >> 1  ; 
    m_period[FAST2] = period >> 2 ; 
    m_period[FAST4] = period >> 3 ; 
    m_period[FAST8] = period >> 4  ; 
    m_period[FAST16] = period >> 5 ; 
    m_period[FAST32] = period >> 6 ; 
    m_period[FAST64] = period >> 7 ;     //  128 >> 7 = 1 : so period must be at least 128 ?

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
    m_fractions[FAST8] = make_fractions(m_period[FAST8]) ;
    m_fractions[FAST16] = make_fractions(m_period[FAST16]) ;
    m_fractions[FAST32] = make_fractions(m_period[FAST32]) ;
    m_fractions[FAST64] = make_fractions(m_period[FAST64]) ;

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
    m_cmd[FAST8] = "TA" ; 
    m_cmd[FAST16] = "TB" ; 
    m_cmd[FAST32] = "TC" ; 
    m_cmd[FAST64] = "TD" ; 

}

/**
Assuming base period 128 

    -5 T1 SLOW32  128*32      4096
    -4 T2 SLOW16  128*16      2048
    -3 T3 SLOW8   128*8       1024
    -2 T4 SLOW4   128*4        512
    -1 T5 SLOW2   128*2        256
     0 T6 NORM    128          128
     1 T7 FAST    128/2         64
     2 T8 FAST2   128/4         32
     3 T9 FAST4   128/8         16
     4 TA FAST8   128/16         8
     5 TB FAST16  128/32         4
     6 TC FAST32  128/64         2
     7 TD FAST64  128/128        1

**/


Animator::Mode_t Animator::getMode() const 
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
      
    int mode = ( cmd[1] > '0' && cmd[1] <= '9' ) ? (int)cmd[1] - (int)'0' : (int)cmd[1] - (int)'A' + 10 ; 

    LOG(info) << " cmd " << cmd << " mode " << mode ; 

    if( mode > -1 && mode < NUM_MODE ) 
    {
         setMode((Mode_t)mode) ; 
    }
    else
    {
         LOG(info) << cmd ; 
         setMode(OFF); 
         home();
    }
}

void Animator::nextMode(unsigned int modifiers)
{
    if(modifiers & OpticksConst::e_shift) m_increment = -m_increment ;

    bool option = 0 != (modifiers & OpticksConst::e_option) ;    
    bool control = 0 != (modifiers & OpticksConst::e_control) ;    
    //bool command = 0 != (modifiers & OpticksConst::e_command) ;    


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

unsigned Animator::getBasePeriod() const 
{
    return m_base_period ; 
}
unsigned Animator::getPeriod() const 
{
    return m_period[m_mode] ;    // with base period 128 in FAST64 mode this period becomes 1
}




bool Animator::step(bool& bump, unsigned& cmd_index, unsigned& cmd_offset )
{
    bool st = step(bump) ; 
    if(!st) return st ; 

    unsigned period = m_period[m_mode] ;    // with period 128 in FAST64 mode this period becomes 1, so no opportunity for "sub-period" cmds
    if( period > m_cmd_slots )
    {      
        m_cmd_tranche = period/m_cmd_slots ; //  NB keep animator_period a suitable power of two, such as 128
        m_cmd_index = m_index/m_cmd_tranche ;  
        assert( m_cmd_index < m_cmd_slots ) ;    
        m_cmd_offset = m_index - m_cmd_index*m_cmd_tranche ;
        assert( m_cmd_offset < m_cmd_tranche ) ;
        cmd_index = m_cmd_index ; 
        cmd_offset = m_cmd_offset ;     
    }
    else
    {
        cmd_index = 0 ; 
        cmd_offset = 0 ; 
    }

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
        case FAST8:mode = FAST8_ ; break ; 
        case FAST16:mode = FAST16_ ; break ; 
        case FAST32:mode = FAST32_ ; break ; 
        case FAST64:mode = FAST64_ ; break ; 
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



