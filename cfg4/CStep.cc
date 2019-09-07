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

#include "CFG4_BODY.hh"

#include "G4SystemOfUnits.hh"
#include "G4StepPoint.hh"
#include "G4Step.hh"

#include "OpticksQuadrant.h"
#include "CStep.hh"

const G4Step* CStep::getStep() const 
{
    return m_step ; 
}
unsigned int CStep::getStepId() const 
{
    return m_step_id ; 
}



unsigned CStep::PreQuadrant(const G4Step* step) // static 
{
    const G4StepPoint* pre  = step->GetPreStepPoint() ; 
    const G4ThreeVector& pos = pre->GetPosition();
    return 
                  (  pos.x() > 0.f ? unsigned(QX) : 0u ) 
                   |   
                  (  pos.y() > 0.f ? unsigned(QY) : 0u ) 
                   |   
                  (  pos.z() > 0.f ? unsigned(QZ) : 0u )
                  ;   
}



double CStep::PreGlobalTime(const G4Step* step) // static
{
    const G4StepPoint* point  = step->GetPreStepPoint() ; 
    return point ? point->GetGlobalTime()/ns : -1 ;
}
double CStep::PostGlobalTime(const G4Step* step) // static
{
    const G4StepPoint* point  = step->GetPostStepPoint() ; 
    return point ? point->GetGlobalTime()/ns : -1 ;
}


const G4Material* CStep::PreMaterial( const G4Step* step) // static
{
    const G4StepPoint* pre  = step->GetPreStepPoint() ; 
    const G4Material* preMat  = pre->GetMaterial() ;
    return preMat ; 
}

const G4Material* CStep::PostMaterial( const G4Step* step) // static
{
    const G4StepPoint* post = step->GetPostStepPoint() ; 
    const G4Material* postMat  = post->GetMaterial() ;
    return postMat ; 
}



CStep::CStep(const G4Step* step, unsigned int step_id) 
   :
   m_step(NULL),
   m_step_id(step_id)
{
   m_step = new G4Step(*step) ;
}

CStep::~CStep()
{
   delete m_step ; 
}


