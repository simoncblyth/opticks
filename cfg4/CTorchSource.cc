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
// based on /usr/local/env/g4/geant4.10.02/source/event/include/G4SingleParticleSource.hh 
#include <cmath>
#include <sstream>

// npy-
#include "NPY.hpp"
#include "NStep.hpp"
#include "TorchStepNPY.hpp"
#include "GLMFormat.hpp"

#include "Opticks.hh"
#include "OpticksFlags.hh"
#include "OpticksGenstep.hh"

// cfg4-
#include "CTorchSource.hh"
#include "CEventInfo.hh"
#include "CRecorder.hh"

// g4-
#include "G4PrimaryParticle.hh"
#include "G4Event.hh"

#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"

#include "Randomize.hh"

#include "G4TrackingManager.hh"
#include "G4Track.hh"

#include "G4SPSPosDistribution.hh"
#include "G4SPSAngDistribution.hh"
#include "G4SPSEneDistribution.hh"
#include "G4SPSRandomGenerator.hh"

#include "PLOG.hh"

const plog::Severity CTorchSource::LEVEL = PLOG::EnvLevel("CTorchSource", "DEBUG") ; 

void CTorchSource::setVerbosity(int verbosity) 
{
	m_verbosity = verbosity ;
	m_posGen->SetVerbosity(verbosity);
	m_angGen->SetVerbosity(verbosity);
	m_eneGen->SetVerbosity(verbosity);
}

CTorchSource::CTorchSource(Opticks* ok, TorchStepNPY* torch, unsigned verbosity)  
    :
    CSource(ok),
    m_torch(torch),
    m_onestep(torch->getOneStep()),
    m_torchdbg(ok->isDbgTorch()),
    m_verbosity(verbosity),
    m_num_photons_total(m_torch->getNumPhotons()),
    m_num_photons_per_g4event(m_torch->getNumPhotonsPerG4Event()),
    m_num_photons( m_num_photons_total < m_num_photons_per_g4event ? m_num_photons_total : m_num_photons_per_g4event ),
    m_posGen(NULL),
    m_angGen(NULL),
    m_eneGen(NULL),
    m_ranGen(NULL),
    m_primary(NPY<float>::make(0,4,4))
{
    init();
}

void CTorchSource::init()
{
	m_ranGen = new G4SPSRandomGenerator();

	m_posGen = new G4SPSPosDistribution();
	m_posGen->SetBiasRndm(m_ranGen);

	m_angGen = new G4SPSAngDistribution();
	m_angGen->SetPosDistribution(m_posGen);
	m_angGen->SetBiasRndm(m_ranGen);

	m_eneGen = new G4SPSEneDistribution();
	m_eneGen->SetBiasRndm(m_ranGen);

    configure();
}

CTorchSource::~CTorchSource() 
{
	delete m_ranGen;
	delete m_posGen;
	delete m_angGen;
	delete m_eneGen;

}


/**
CTorchSource::configure
------------------------

Translate the Opticks TorchStepNPY into Geant4 generator photon source

**/

std::string CTorchSource::desc() const 
{
     std::stringstream ss ; 
     ss << "CTorchSource"
        << " num_photons_total " << m_num_photons_total
        << " num_photons_per_g4event " << m_num_photons_per_g4event
        << " num_photons " << m_num_photons 
        ;   
     return ss.str(); 
}


void CTorchSource::configure()
{
    if(m_torchdbg)
    {
        m_torch->Summary("[--torchdbg] CTorchSource::configure");
        LOG(info) << m_torch->description();
        LOG(info) << desc() ; 
    }


    float w = m_onestep->getWavelength() ; 
    if(w > 0.f)
    {
        G4double wavelength = w*nm ; 
        G4double energy = h_Planck*c_light/wavelength ;
        m_eneGen->SetEnergyDisType("Mono");
        m_eneGen->SetMonoEnergy(energy);
    }
    else
    {
        assert(0 && "only mono supported"); 
    }


    float _t       = m_onestep->getTime();
    glm::vec3 _pos = m_onestep->getPosition();
    glm::vec3 _dir = m_onestep->getDirection();
    glm::vec3 _pol = m_onestep->getPolarization() ; 
    float _radius  = m_onestep->getRadius();
    glm::vec4 _zeaz = m_onestep->getZenithAzimuth();

    LOG(fatal) << "CTorchSource::configure"
               << " _t " << _t 
               << " _radius " << _radius
               << " _pos " << gformat(_pos) 
               << " _dir " << gformat(_dir) 
               << " _zeaz " << gformat(_zeaz)
               << " _pol " << gformat(_pol)
                ; 


    G4ThreeVector pos(_pos.x*mm,_pos.y*mm,_pos.z*mm);

    G4ThreeVector cen(pos);

    // TODO: from config? need more state as the pol holds surfaceNormal ?
    //G4ThreeVector pol(1.,0.,0.); 
    //SetParticlePolarization(pol); // reset later for the custom configs 

    bool incidentSphere = m_torch->isIncidentSphere() ;
    bool disc           = m_torch->isDisc() ;
    bool discLin        = m_torch->isDiscLinear() ;
    bool ring           = m_torch->isRing() ;
    bool point          = m_torch->isPoint() ;
    bool sphere         = m_torch->isSphere() ;
    bool reflTest       = m_torch->isReflTest() ;

    // for sanity test geometries use standard X Y for 

    G4ThreeVector posX(1,0.,0.);
    G4ThreeVector posY(0,1.,0.);
    m_posGen->SetPosRot1(posX);
    m_posGen->SetPosRot2(posY);

    if(point )
    {
        m_posGen->SetPosDisType("Point");
        m_posGen->SetCentreCoords(cen);

        G4ThreeVector dir(_dir.x,_dir.y,_dir.z);
        m_angGen->SetParticleMomentumDirection(dir);

    }
    else if( sphere )
    {
        m_posGen->SetPosDisType("Point");
        m_posGen->SetCentreCoords(cen);

        m_angGen->SetAngDistType("iso");
    
    }
    else if(incidentSphere || discLin || disc || ring)  // used with "rainbow" geometry 
    {
        m_posGen->SetPosDisType("Plane");

        float ze_x = _zeaz.x ; 
        float ze_y = _zeaz.y ; 
       
        if(ze_x == 0.f )
        {
            m_posGen->SetPosDisShape("Circle");
            m_posGen->SetRadius(_radius*mm);
            LOG(info) << "CTorchSource::configure posGen Circle"
                      << " radius " << m_posGen->GetRadius()
                      << " ze_x " << ze_x
                      << " ze_y " << ze_y
                      ;
        }
        else
        {
            m_posGen->SetPosDisShape("Annulus");
            float radius0 = float(_radius*mm*ze_x) ;

            m_posGen->SetRadius0(radius0);
            m_posGen->SetRadius( _radius*mm*ze_y);

            LOG(info) << "CTorchSource::configure posGen Annulus"
                      << " radius " << m_posGen->GetRadius()
                      << " radius0 " << radius0
                      << " ze_x " << ze_x
                      << " ze_y " << ze_y
                      ;

        }
 
        m_posGen->SetCentreCoords(cen);

        m_angGen->SetAngDistType("planar");
        G4ThreeVector dir(_dir.x,_dir.y,_dir.z);
        m_angGen->SetParticleMomentumDirection(dir);

    } 
    else if( reflTest ) 
    {
        m_posGen->SetPosDisType("Surface");
        m_posGen->SetPosDisShape("Sphere");
        m_posGen->SetRadius(_radius*mm);
        m_posGen->SetCentreCoords(cen);

        m_angGen->SetAngDistType("focused");
        m_angGen->SetFocusPoint(cen); 
    }
    else
    {
        LOG(warning) << "CTorchSource::configure mode not handled, default position/direction generators will be used " ; 
    }

    //for(unsigned int i=0 ; i < 10 ; i++) G4cout << Format(posGen->GenerateOne(), "posGen", 10) << G4endl ; 
    //for(unsigned int i=0 ; i < 10 ; i++) G4cout << Format(angGen->GenerateOne(), "angGen") << G4endl ; 

}




void CTorchSource::GeneratePrimaryVertex(G4Event *event) 
{
    LOG(LEVEL)
        << " NumberOfParticlesToBeGenerated: "
		<< m_num_photons
        ;

    //unsigned event_gencode = TORCH ;   
    unsigned event_gencode = OpticksGenstep_TORCH ; 
    event->SetUserInformation( new CEventInfo(event_gencode)) ;

    unsigned photon_flag = OpticksGenstep::GenstepToPhotonFlag(event_gencode); 

   LOG(info)
        << " event_gencode " << event_gencode
        << " : " << OpticksFlags::Flag(photon_flag)
        ; 


    float _t = m_onestep->getTime();
    G4double time = _t*ns ; 

    glm::vec3 pol = m_onestep->getPolarization() ;
    G4ThreeVector fixpol(pol.x, pol.y, pol.z);   

    G4ParticleDefinition* definition = G4OpticalPhoton::Definition() ;  

	for (unsigned i = 0; i < m_num_photons ; i++) 
    {
	    G4ThreeVector position = m_posGen->GenerateOne();

        G4PrimaryVertex* vertex = new G4PrimaryVertex(position, time);

		G4ParticleMomentum direction = m_angGen->GenerateOne();

		G4double kineticEnergy = m_eneGen->GenerateOne(definition);

        if(m_torchdbg && i < 10 && m_verbosity > 5) 
        LOG(info) << "CTorchSource::GeneratePrimaryVertex"
                  << " i " << std::setw(6) << i 
                  << " posx " << position.x()
                  << " posy " << position.y()
                  << " posz " << position.z()
                  << " dirx " << direction.x()
                  << " diry " << direction.y()
                  << " dirz " << direction.z()
                  ;

		G4PrimaryParticle* particle = new G4PrimaryParticle(definition);
		particle->SetKineticEnergy( kineticEnergy );
		particle->SetMomentumDirection( direction );

        if(m_torch->isIncidentSphere())
        {
            // custom S/P-polarization specific to "rainbow geometry" 
            // (planar disc of photons incident on sphere with same radius as disc)
            //
            G4ThreeVector tangent(-position.y(),  position.x(), 0. );  // anti-clockwise tangent
            G4ThreeVector radial(  position.x(),  position.y(), 0. ); 
            if(m_torch->isSPolarized())
            { 
		       particle->SetPolarization(tangent.unit()); 
            }
            else if(m_torch->isPPolarized())
            {
		       particle->SetPolarization(radial.unit());  
            }
            else
            {
		       particle->SetPolarization(pol.x, pol.y, pol.z );
            }

        }
        else if(m_torch->isDiscLinear() || m_torch->isDisc())
        {
            // match the adhoc (sinPhi, -cosPhi, 0) polarization from optixrap-/cu/torchstep.h:generate_torch_photon
            G4ThreeVector tangent(position.y(),  -position.x(), 0. );  
		    particle->SetPolarization(tangent.unit()); 
        }
        else if(m_torch->isPoint())
        {
            G4ThreeVector tangent(position.y(),  -position.x(), 0. );  
		    particle->SetPolarization(tangent.unit()); 
        }
        else if(m_torch->isSphere())
        {
            // see  cu/torchstep.h
            G4ThreeVector pol0(1,0,0); // <-- this direction needs to not be parallel to mom dir ?
            G4ThreeVector perp = pol0 - pol0.dot(direction) * direction;  
            // subtract vector component in the momentum direction, to yield perpendicular polarization
		    particle->SetPolarization(perp.unit()); 
        } 
        else
        {
		    particle->SetPolarization(pol.x, pol.y, pol.z );
        }  


        if(m_torch->isFixPolarized())
        {
             //LOG(fatal) << "CTorchSource::GeneratePrimaryVertex" 
             //          << " fixpol override "
             //          << gformat(polarization)
             //          ; 
		     particle->SetPolarization(fixpol.unit());
        }


		if (m_verbosity > 2) 
        {
			LOG(info) << "Particle name: "
					  << definition->GetParticleName() 
                      << " verbosity " << m_verbosity
                      << " kineticEnergy " << kineticEnergy
                      << " position " << position
                      << " direction " << direction
                      ;
		}
		// Set bweight equal to the multiple of all non-zero weights
		G4double weight = m_eneGen->GetWeight()*m_ranGen->GetBiasWeight();
		particle->SetWeight(weight);

		vertex->SetPrimary(particle);
        event->AddPrimaryVertex(vertex);

        collectPrimaryVertex(vertex);
	}
}


