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
class G4StepNPY ; 

#include "NPY_API_EXPORT.hh"

class NPY_API TrivialCheckNPY {

       enum {
           IS_UINDEX,
           IS_UINDEX_SCALED,
           IS_UCONSTANT,
           IS_UCONSTANT_SCALED
       };

   public:  
       static bool IsApplicable( char entryCode);
       TrivialCheckNPY(NPY<float>* photons, NPY<float>* gensteps, char entryCode);
       int checkItemValue(unsigned istep, NPY<float>* npy, unsigned i0, unsigned i1, unsigned jj, unsigned kk, const char* label, int expect, int constant=0, int scale=0 );
   public:  
       void dump(const char* msg="TrivialCheckNPY::dump");
       int check(const char* msg);
  private:
        void checkGensteps(NPY<float>* gs);
        int checkPhotons(unsigned istep, NPY<float>* photons, unsigned i0, unsigned i1, unsigned gencode, unsigned numPhotons);
  private:
        char         m_entryCode ; 
        NPY<float>*  m_photons ; 
        NPY<float>*  m_gensteps ; 
        G4StepNPY*   m_g4step ; 
};



