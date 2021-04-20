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

#pragma once

/**
Animator
==========

Changes the value of \*m_target at each Animator::step

**/



// TODO: try to support live changing of the range 

//#define ANIMATOR_DEBUG 1

#include <string>
#include "plog/Severity.h"
#include "OKCORE_API_EXPORT.hh"

class OKCORE_API Animator {
    public:
        friend class GUI ; 
    public:
        static const plog::Severity LEVEL ; 
        static const int period_low ; 
        static const int period_high ; 

        static const int LEVEL_MIN ; 
        static const int LEVEL_MAX ; 

        static const char* OFF_ ; 
        static const char* SLOW32_ ; 
        static const char* SLOW16_ ; 
        static const char* SLOW8_ ; 
        static const char* SLOW4_ ; 
        static const char* SLOW2_ ; 
        static const char* NORM_ ; 
        static const char* FAST_ ; 
        static const char* FAST2_ ; 
        static const char* FAST4_ ; 
        static const char* FAST8_ ; 
        static const char* FAST16_ ; 
        static const char* FAST32_ ; 
        static const char* FAST64_ ; 

        //                     -5     -4     -3     -2      -1     0     1      2     3       4      5       6      7 
        //              T0     T1     T2     T3     T4      T5    T6    T7     T8    T9      TA     TB      TC     TD
        typedef enum {  OFF, SLOW32, SLOW16, SLOW8, SLOW4, SLOW2, NORM, FAST, FAST2, FAST4, FAST8, FAST16, FAST32, FAST64, NUM_MODE } Mode_t ;

        static Mode_t Mode(int imode); 
        static Mode_t Mode(const char* name); 

        Animator(float* target, unsigned int period, float low=0.f, float high=1.f, const char* label="AnimatorLabel");

        void setModeRestrict(Mode_t restrict_);
        bool isModeChanged(Mode_t prior);    
        void modeTransition(float fraction);

        bool isSlowEnabled();
        bool isNormEnabled();
        bool isFastEnabled();

        void home();
        void reset();

        void setModeForPeriod(unsigned period); 
        unsigned getBasePeriod() const ;
        unsigned getPeriod() const ;

        bool step(bool& bump); 
        bool step(bool& bump, unsigned& cmd_index, unsigned& cmd_offset); 
        void Summary(const char* msg);
        void scrub_to(float x, float y, float dx, float dy); // Interactor:K scrub_mode

        float* getTarget(); 
        float getFraction();
        float getFractionFromTarget();
        float getLow(); 
        float getHigh(); 
        bool isActive();

        Mode_t getMode() const ;


        int* getModePtr();
        unsigned int getNumMode();

        void setMode( int level );
        void setMode( const char* name );
        void setMode( Mode_t mode);

        void nextMode(unsigned int modifiers);
        void commandMode(const char* cmd);

        const char* getModeName() const ;
        const char* getModeCmd() const ;
        std::string desc() const ;

        char* description();

    private:
       // used for scrubbing
        void          setTargetValue(float value);
        void          setFraction(float f);
    private:
        void          setTarget(float* target); // qty to be stepped
        unsigned int  getIndex();
        float         getValue();
        float         getFractionForValue(float value);
        unsigned int  find_closest_index(float f);
        bool          isBump();

    private:
        float* make_fractions(unsigned int num);

    private:
        Mode_t       m_mode ; 
        Mode_t       m_restrict ; 
        unsigned     m_period[NUM_MODE] ; 
        unsigned     m_base_period ; 
        float        m_low ; 
        float        m_high ; 
        const char*  m_label ; 
        float*       m_fractions[NUM_MODE] ; 
        const char*  m_cmd[NUM_MODE] ; 

        unsigned int m_count ; 
        unsigned int m_index ; 
        char         m_desc[32] ; 
        float*       m_target ; 
        int          m_increment ; 

        unsigned     m_cmd_slots ; 
        unsigned     m_cmd_index ; 
        unsigned     m_cmd_offset ; 
        unsigned     m_cmd_tranche ; 
};



