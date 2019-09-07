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

// g4-
class G4VFigureFileMaker ;
class G4VRTScanner ;
class G4TheRayTracer ; 

// okc-
class Opticks ; 
class Composition ; 

// okg-
class OpticksHub ; 

// cfg4-
class CG4 ; 

#include "CFG4_API_EXPORT.hh"

/**
CRayTracer
============

Canonical m_rt instance is ctor resident of CG4. 

**/


class CFG4_API CRayTracer
{
   public:
        CRayTracer(CG4* g4);
        void snap() const ;
   private:
        CG4*         m_g4 ; 
        Opticks*     m_ok ;
        OpticksHub*  m_hub ; 
        Composition* m_composition ; 

        G4VFigureFileMaker* m_figmaker ; 
        G4VRTScanner*       m_scanner ; 
        G4TheRayTracer*     m_tracer ; 
  
};

    
