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
#include "G4MaterialPropertyVector.hh"


#include "CFG4_API_EXPORT.hh"
class CFG4_API CVec 
{
    public: 
         static std::string Digest(CVec* vec);  
         static std::string Digest(G4MaterialPropertyVector* vec);
         static CVec* MakeDummy(size_t n ); 
    public:
         CVec(G4MaterialPropertyVector* vec) ; 
         std::string  digest();
         G4MaterialPropertyVector* getVec(); 
         float getValue(float nm);
         void  dump(const char* msg="CVec::dump", float lo=60.f, float hi=720.f, float step=20.f);
    private:
         G4MaterialPropertyVector* m_vec ; 

};
