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

#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

class Opticks ; 
template <typename T> class OpticksCfg ;

class OKCORE_API OpticksEntry 
{
   private:
        static const char* GENERATE_ ; 
        static const char* TRIVIAL_ ; 
        static const char* NOTHING_ ; 
        static const char* SEEDTEST_ ; 
        static const char* TRACETEST_ ; 
        static const char* ZRNGTEST_ ; 
        static const char* UNKNOWN_ ; 
   public:
        static const char*  Name(char code);
        static char CodeFromConfig(OpticksCfg<Opticks>* cfg);  
   public:
        OpticksEntry(unsigned index, char code);
   public:
        unsigned      getIndex();
        const char*   getName();
        std::string   description() const ;
        std::string   desc() const ;
        bool          isTrivial();
        bool          isNothing();
        bool          isTraceTest();
   private:
        unsigned             m_index ; 
        char                 m_code ; 

};

#include "OKCORE_TAIL.hh"


