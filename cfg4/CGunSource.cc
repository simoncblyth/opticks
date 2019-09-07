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
#include <cassert>

#include "NGunConfig.hpp"

#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "G4PrimaryParticle.hh"
#include "G4Event.hh"
#include "G4ParticleMomentum.hh"

#include "CParticleDefinition.hh"
#include "CRecorder.hh"
#include "CGunSource.hh"

#include "PLOG.hh"

CGunSource::CGunSource(Opticks* ok)  
    :
    CSource(ok),
    m_config(NULL)
{
}

CGunSource::~CGunSource() 
{
}

void CGunSource::configure(NGunConfig* gc)
{
    m_config = gc ; 

    LOG(info) << "CGunSource::configure" ; 
    m_config->Summary("CGunSource::configure");
}

void CGunSource::GeneratePrimaryVertex(G4Event *event) 
{
    LOG(fatal) << "CGunSource::GeneratePrimaryVertex" ;

    glm::vec3 pos = m_config->getPosition();
    G4ThreeVector position( pos.x*mm, pos.y*mm, pos.z*mm );
    G4double time =  m_config->getTime()*ns ; 

    G4PrimaryVertex* vertex = new G4PrimaryVertex(position, time);

    G4double kineticEnergy = m_config->getEnergy()*MeV ;
    glm::vec3 dir = m_config->getDirection();   
    glm::vec3 pol = m_config->getPolarization();

    G4ParticleDefinition* definition = CParticleDefinition::Find(m_config->getParticle()) ; 
    G4PrimaryParticle* primary = new G4PrimaryParticle(definition);
    primary->SetKineticEnergy( kineticEnergy);

    G4ThreeVector direction(dir.x, dir.y, dir.z) ;  
    primary->SetMomentumDirection( direction );
	primary->SetPolarization(pol.x, pol.y, pol.z ); 

    vertex->SetPrimary(primary);
    event->AddPrimaryVertex(vertex);
 
    LOG(info) << "CGunSource::GeneratePrimaryVertex" 
              << " time " << time 
              << " position.x " << position.x() 
              << " position.y " << position.y() 
              << " position.z " << position.z() 
              ; 

    //m_recorder->RecordPrimaryVertex(vertex);
}


