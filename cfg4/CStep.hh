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


class G4Step ; 
class G4Material ; 

/**
CStep
======

* *CStep* ctor copies the argument G4Step 
  and holds the pointer to the copy together with step_id 

**/

#include "CFG4_API_EXPORT.hh"
class CFG4_API CStep {
   public:
       static unsigned PreQuadrant(const G4Step* step);
       static double PreGlobalTime(const G4Step* step);
       static double PostGlobalTime(const G4Step* step);
       static const G4Material* PreMaterial( const G4Step* step) ;
       static const G4Material* PostMaterial( const G4Step* step) ;

       CStep(const G4Step* step, unsigned int step_id);
       virtual ~CStep();
       const G4Step* getStep() const ;  
       unsigned int  getStepId() const ; 
   private:
       const G4Step* m_step ; 
       unsigned int  m_step_id ; 
};


