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

#include "G4VUserPrimaryGeneratorAction.hh"
#include "globals.hh"

class G4VPrimaryGenerator ;
class G4Event;
class CSource ; 


#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

/**
CPrimaryGeneratorAction
=========================

Main method *GeneratePrimaries* invoked by Geant4 beamOn within CG4::propagate
invokes CSource::GeneratePrimaryVertex(G4Event*)

**/


class CPrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction
{
  public:
    CPrimaryGeneratorAction(CSource* generator);
    virtual ~CPrimaryGeneratorAction();
  public:
    virtual void GeneratePrimaries(G4Event*);
  private:
    CSource*  m_source ;

};
#include "CFG4_TAIL.hh"

