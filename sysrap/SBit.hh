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

#include "SYSRAP_API_EXPORT.hh"
#include "SYSRAP_HEAD.hh"

#include <string>

/**
SBit
=====


**/

class SYSRAP_API SBit {
    public:
         // ffs returns 1-based index of rightmost set bit, see man ffs 
        static int ffs(int msk);
        static long long ffsll(long long msk);

        static unsigned long long count_nibbles(unsigned long long x); 

        static bool HasOneSetBit(int x); 

        template <typename T>
        static std::string BinString(T v); 

        template <typename T>
        static std::string HexString(T v); 

        static unsigned long long FromBinString(const char* binstr ) ; 
        static unsigned long long FromHexString(const char* hexstr ) ; 
        static unsigned long long FromDecString(const char* decstr ) ; 
        static unsigned long long FromString(const char* str ) ;   // if first two char are 0x/0b the string is interpreted as hex/binary otherwise decimal  

}; 

#include "SYSRAP_TAIL.hh"


