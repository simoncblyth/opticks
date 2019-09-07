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
TCURAND
==========

High level interface for dynamic GPU generation of curand 
random numbers, exposing functionality from TRngBuf  

**/

#include "THRAP_API_EXPORT.hh" 
#include "plog/Severity.h"

template <typename T> class NPY ; 
template <typename T> class TCURANDImp ; 

template<typename T>
class THRAP_API TCURAND
{
        static const plog::Severity LEVEL ; 
    public:
        TCURAND( unsigned ni, unsigned nj, unsigned nk);  
        void     setIBase(unsigned ibase); 
        unsigned getIBase() const ; 
        NPY<T>*  getArray() const ; 
    private:
        void     generate();       // called by setIBase, updates contents of array 
    private:
        TCURANDImp<T>*  m_imp ; 

}; 



