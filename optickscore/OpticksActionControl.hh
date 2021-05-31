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
#include <string>
#include <vector>

#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

/**
OpticksActionControl
======================

Used to label gensteps via::

    272 void NPYBase::addActionControl(unsigned long long control)
    273 {
    274     m_action_control |= control ;
    275 }
    276 void NPYBase::setActionControl(unsigned long long control)
    277 {
    278     m_action_control = control ;
    279 }

**/


class OKCORE_API OpticksActionControl {
    public:
        enum {
                GS_LOADED      = 0x1 << 1,
                GS_FABRICATED  = 0x1 << 2,
                GS_TRANSLATED  = 0x1 << 3,
                GS_TORCH       = 0x1 << 4,
                GS_LEGACY      = 0x1 << 5,
                GS_EMBEDDED    = 0x1 << 6,
                GS_EMITSOURCE  = 0x1 << 7
             };  
    public:
        static const char* GS_LOADED_  ; 
        static const char* GS_FABRICATED_ ; 
        static const char* GS_TRANSLATED_ ; 
        static const char* GS_TORCH_ ; 
        static const char* GS_LEGACY_ ;
        static const char* GS_EMBEDDED_;
        static const char* GS_EMITSOURCE_;
    public:
        static std::string Desc(unsigned long long ctrl);
        static unsigned long long Parse(const char* ctrl, char delim=',');
        static unsigned long long ParseTag(const char* ctrl);
        static bool isSet(unsigned long long ctrl, const char* mask);
        static unsigned NumSet(unsigned long long ctrl); 

        static std::vector<const char*> Tags();
    public:
        OpticksActionControl(unsigned long long* ctrl); 
        void add(const char* mask);
        unsigned numSet() const ;  
        bool isSet(const char* mask) const;
        bool operator()(const char* mask) const;
        std::string desc(const char* msg=nullptr) const;
    private:
         unsigned long long* m_ctrl ; 

};

#include "OKCORE_TAIL.hh"


