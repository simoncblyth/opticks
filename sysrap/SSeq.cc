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



#include <cassert>
#include "SSeq.hh"


template <typename T>
SSeq<T>::SSeq(T seq_) 
   : 
   seq(seq_), 
   zero(0ull) 
{} ;

template <typename T>
T SSeq<T>::msn()   // most significant nibble
{
    unsigned nnib = sizeof(T)*2 ; 
    for(unsigned i=0 ; i < nnib ; i++)
    {
        T f = nibble(nnib-1-i) ; 
        if( f == zero ) continue ; 
        return f ; 
    } 
    return zero ; 
}


template <typename T>
T SSeq<T>::nibble(unsigned i)
{
    return (seq >> i*4) & T(0xF) ; 
}


template struct SSeq<unsigned>;
template struct SSeq<unsigned long long>;


