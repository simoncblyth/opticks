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

#include "NGLM.hpp"

// okc-
#include "OpticksGenstep.hh"
#include "OpticksPhoton.h"
#include "OpticksPhoton.hh"
#include "Opticks.hh"

// g4-
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"

#include "G4ParticleChange.hh"
#include "G4OpticalPhoton.hh"
#include "G4PrimaryParticle.hh"
#include "G4Event.hh"
#include "G4ParticleMomentum.hh"
#include "G4ParticleTable.hh"
#include "G4ThreeVector.hh"

// cfg4-
#include "CGenstepSource.hh"
#include "CCerenkovGenerator.hh"
#include "CGenstepSource.hh"
#include "C4PhotonCollector.hh"
#include "CEventInfo.hh"

#include "STranche.hh"
#include "PLOG.hh"

unsigned CGenstepSource::getNumG4Event() const { return m_tranche->num_tranche ; }
unsigned CGenstepSource::getNumGenstepsPerG4Event() const { return m_num_genstep_per_g4event ; }

CGenstepSource::CGenstepSource(Opticks* ok, NPY<float>* gs)  
    :
    CSource(ok),
    m_gs(new OpticksGenstep(gs)),
    m_num_genstep(m_gs->getNumGensteps()),
    m_num_genstep_per_g4event(1),
    m_tranche(new STranche(m_num_genstep,m_num_genstep_per_g4event)),
    m_idx(0),
    m_generate_count(0),
    m_photon_collector(new C4PhotonCollector) 
{
    init();
}
void CGenstepSource::init()
{
}

CGenstepSource::~CGenstepSource() 
{
}


NPY<float>* CGenstepSource::getSourcePhotons() const
{
    return m_photon_collector->getPhoton() ; 
}



void CGenstepSource::GeneratePrimaryVertex(G4Event *event) 
{
    std::vector<unsigned> gencodes ; 

    for(unsigned i=0 ; i < m_num_genstep_per_g4event ; i++ )
    {
        if( m_idx == m_num_genstep ) break ;    // all gensteps done

        unsigned gencode = getCurrentGencode() ;

        // collect unique gencodes  
        if(std::find(gencodes.begin(), gencodes.end(), gencode ) == gencodes.end() ) gencodes.push_back(gencode) ;  

        G4VParticleChange* pc = generatePhotonsFromOneGenstep() ; 

        addPrimaryVertices( event, pc ); 
    }

    assert( gencodes.size() == 1 && "expecting only a single type of gencode within each G4Event" ); 

    unsigned event_gencode = gencodes[0] ; 

    LOG(info)
        << " event_gencode " << event_gencode
        << " : " << OpticksPhoton::Flag(event_gencode)
        ; 

    event->SetUserInformation( new CEventInfo(event_gencode)) ;   

    m_generate_count += 1 ; 
}



unsigned CGenstepSource::getCurrentGencode() const 
{
    unsigned gencode = m_gs->getGencode(m_idx) ; 
    return gencode ; 
}


/**
CGenstepSource::generatePhotonsFromOneGenstep
----------------------------------------------

Notice that genstep arrays can contain mixed types of gensteps, BUT that
each individual genstep is always of one particular type.

SUSPECT confusion here between genstep codes and photon codes.

**/

G4VParticleChange* CGenstepSource::generatePhotonsFromOneGenstep()
{
    assert( m_idx < m_num_genstep ); 
    unsigned gencode = getCurrentGencode() ; 
    LOG(info)
        << " gencode " << gencode
        << " OpticksPhoton::Flag(gencode) " << OpticksPhoton::Flag(gencode)
        ; 

    G4VParticleChange* pc = NULL ; 

    switch( gencode )
    { 
        case CERENKOV:      pc = CCerenkovGenerator::GeneratePhotonsFromGenstep(m_gs,m_idx) ; break ; 
        case SCINTILLATION: pc = NULL                                                       ; break ;  
        default:            pc = NULL ; 
    }

    if(!pc)
        LOG(fatal) 
            << " failed to generate for "
            << " gencode " << gencode
            << " flag " << OpticksPhoton::Flag(gencode) 
            ; 
   
    assert( pc ); 

    m_photon_collector->collectSecondaryPhotons( pc, m_idx );  // "Secondary" : but this makes them primary 

    m_idx += 1 ; 

    return pc ; 
}


void CGenstepSource::addPrimaryVertices(G4Event *event,  const G4VParticleChange* pc) const 
{

    unsigned ngp = m_ok->getNumGenPhoton();
    if(ngp > 0u)
    {
        LOG(fatal) << "restricting photons via --gindex option, ngp: " << ngp ;  
    }


    G4int numberOfSecondaries = pc->GetNumberOfSecondaries(); 
    LOG(info) << " numberOfSecondaries " << numberOfSecondaries ; 

    for( G4int i=0 ; i < numberOfSecondaries ; i++)
    {
        if( ngp > 0u && !m_ok->isGenPhoton(i) ) continue ; 


        G4Track* track =  pc->GetSecondary(i) ; 
        const G4ParticleDefinition* definition = track->GetParticleDefinition() ; 
        assert( definition == G4OpticalPhoton::Definition() );  
        const G4DynamicParticle* photon = track->GetDynamicParticle() ;
        const G4ThreeVector& direction = photon->GetMomentumDirection() ;   
        const G4ThreeVector& polarization = photon->GetPolarization() ; 
        G4double kineticEnergy = photon->GetKineticEnergy() ;  
        //G4double wavelength = h_Planck*c_light/kineticEnergy ; 
        const G4ThreeVector& position = track->GetPosition() ;
        G4double time = track->GetGlobalTime() ;
        G4double weight = track->GetWeight() ; 

        // --

        G4PrimaryVertex* vertex = new G4PrimaryVertex(position, time);
		G4PrimaryParticle* particle = new G4PrimaryParticle(definition);
		particle->SetKineticEnergy(kineticEnergy );
		particle->SetMomentumDirection( direction );
        particle->SetPolarization( polarization );
        particle->SetWeight(weight);
		vertex->SetPrimary(particle);
        event->AddPrimaryVertex(vertex);
    }
}




