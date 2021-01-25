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

template <typename T> class NPY ; 
#include "NPY_API_EXPORT.hh"

/**
DummyPhotonsNPY
================





**/

class NPY_API DummyPhotonsNPY 
{
    public:
       static NPY<float>* Make(unsigned num_photons, unsigned hitmask, unsigned modulo=10, unsigned num_quad=4 );  // formerly hitmask was default of:  0x1 << 5  (32)
    private:
       DummyPhotonsNPY(unsigned num_photons, unsigned hitmask, unsigned modulo, unsigned num_quad);
       NPY<float>* getNPY();
       void        init();
    private:
       NPY<float>* m_data    ; 
       unsigned    m_hitmask ; 
       unsigned    m_modulo  ; 
       unsigned    m_num_quad  ; 
};



