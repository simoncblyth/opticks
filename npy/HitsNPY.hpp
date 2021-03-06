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
class NSensorList ; 

/**
HitsNPY
========

This class may be obsolete, using the old sensot approach.

TODO: investigate and eliminate

**/

#include "NPY_API_EXPORT.hh"
class NPY_API HitsNPY {
   public:  
       HitsNPY(NPY<float>* photons, NSensorList* sensorlist); 
   public:
       void debugdump(const char* msg="HitsNPY::debugdump");
   private:
       NPY<float>*                  m_photons ; 
       NSensorList*                 m_sensorlist ; 

};


