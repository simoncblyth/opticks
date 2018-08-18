#include "CFG4_BODY.hh"
// based on /usr/local/env/g4/geant4.10.02/source/event/include/G4SingleParticleSource.hh 
#include <cmath>

// npy-
#include "NPY.hpp"
#include "TorchStepNPY.hpp"
#include "GLMFormat.hpp"

#include "Opticks.hh"

// cfg4-
#include "CTorchSource.hh"
#include "CRecorder.hh"

// g4-
#include "G4AutoLock.hh"
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


CTorchSource::CTorchSource(Opticks* ok, TorchStepNPY* torch, unsigned int verbosity)  
    :
    CSource(ok, verbosity),
    m_torch(torch),
    m_torchdbg(ok->isDbgTorch()),
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
    
    G4MUTEXINIT(m_mutex);
}


CTorchSource::~CTorchSource() 
{
	delete m_ranGen;
	delete m_posGen;
	delete m_angGen;
	delete m_eneGen;

    G4MUTEXDESTROY(m_mutex);
}


/**
CTorchSource::configure
------------------------

Translate the Opticks TorchStepNPY into Geant4 generator photon source


**/


void CTorchSource::configure()
{
    unsigned numPhotons = m_torch->getNumPhotons();
    unsigned numPhotonsPerG4Event = m_torch->getNumPhotonsPerG4Event();
    unsigned n = numPhotons < numPhotonsPerG4Event ? numPhotons : numPhotonsPerG4Event ;

    if(m_torchdbg)
    {
        m_torch->Summary("[--torchdbg] CTorchSource::configure");
        LOG(info) << m_torch->description();

        LOG(info)
           << " numPhotons " << numPhotons 
           << " numPhotonsPerG4Event " << numPhotonsPerG4Event
           << " n " << n 
           ;
    }


    SetNumberOfParticles(n);

    setParticle("opticalphoton");

    float w = m_torch->getWavelength() ; 
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


    float _t       = m_torch->getTime();
    glm::vec3 _pos = m_torch->getPosition();
    glm::vec3 _dir = m_torch->getDirection();
    glm::vec3 _pol = m_torch->getPolarization() ; 
    float _radius  = m_torch->getRadius();
    glm::vec4 _zeaz = m_torch->getZenithAzimuth();

    LOG(fatal) << "CTorchSource::configure"
               << " _t " << _t 
               << " _radius " << _radius
               << " _pos " << gformat(_pos) 
               << " _dir " << gformat(_dir) 
               << " _zeaz " << gformat(_zeaz)
               << " _pol " << gformat(_pol)
                ; 


    SetParticleTime(_t*ns);
    G4ThreeVector pos(_pos.x*mm,_pos.y*mm,_pos.z*mm);
    SetParticlePosition(pos);

    G4ThreeVector cen(pos);

    // TODO: from config? need more state as the pol holds surfaceNormal ?
    G4ThreeVector pol(1.,0.,0.); 
    SetParticlePolarization(pol); // reset later for the custom configs 

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





void CTorchSource::SetVerbosity(int vL) 
{
    G4AutoLock l(&m_mutex);
	m_verbosityLevel = vL;
	m_posGen->SetVerbosity(vL);
	m_angGen->SetVerbosity(vL);
	m_eneGen->SetVerbosity(vL);
}


void CTorchSource::GeneratePrimaryVertex(G4Event *evt) 
{
    assert(m_definition);
    if (m_verbosityLevel > 1)
        LOG(info) << " NumberOfParticlesToBeGenerated: "
				  << m_num 
                  << " verbosityLevel " << m_verbosityLevel 
                   ;

    part_prop_t& pp = m_pp.Get();


    glm::vec3 polarization = m_torch->getPolarization() ;
    G4ThreeVector fixpol(polarization.x, polarization.y, polarization.z);   

    // hmm starting as 0.0000,5005.0000,0.0000 
    //      some frame tranform ???
    //

	//if (m_verbosityLevel > 1)
   /*
    LOG(fatal) << "CTorchSource::GeneratePrimaryVertex"
              << m_torch->description()
              << " fixpol " << fixpol
              << " num " << m_num 
              ;

    */

	for (G4int i = 0; i < m_num; i++) 
    {
	    pp.position = m_posGen->GenerateOne();
        G4PrimaryVertex* vertex = new G4PrimaryVertex(pp.position,m_time);

		pp.momentum_direction = m_angGen->GenerateOne();

		pp.energy = m_eneGen->GenerateOne(m_definition);

        if(m_torchdbg && i < 10 && m_verbosityLevel > 5) 
        LOG(info) << "CTorchSource::GeneratePrimaryVertex"
                  << " i " << std::setw(6) << i 
                  << " posx " << pp.position.x()
                  << " posy " << pp.position.y()
                  << " posz " << pp.position.z()
                  << " dirx " << pp.momentum_direction.x()
                  << " diry " << pp.momentum_direction.y()
                  << " dirz " << pp.momentum_direction.z()
                  ;


		//if (m_verbosityLevel >= 2)
		//	G4cout << "Creating primaries and assigning to vertex" << G4endl;
		// create new primaries and set them to the vertex


        G4double mass = m_definition->GetPDGMass();
        G4double charge = m_definition->GetPDGCharge();

		G4PrimaryParticle* particle = new G4PrimaryParticle(m_definition);
		particle->SetKineticEnergy(pp.energy );
		particle->SetMass( mass );
		particle->SetMomentumDirection( pp.momentum_direction );
		particle->SetCharge( charge );

        if(m_torch->isIncidentSphere())
        {
            // custom S/P-polarization specific to "rainbow geometry" 
            // (planar disc of photons incident on sphere with same radius as disc)
            //
            G4ThreeVector tangent(-pp.position.y(),  pp.position.x(), 0. );  // anti-clockwise tangent
            G4ThreeVector radial(  pp.position.x(),  pp.position.y(), 0. ); 
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
		       particle->SetPolarization(m_polarization.x(), m_polarization.y(), m_polarization.z());
            }

        }
        else if(m_torch->isDiscLinear() || m_torch->isDisc())
        {
            // match the adhoc (sinPhi, -cosPhi, 0) polarization from optixrap-/cu/torchstep.h:generate_torch_photon
            G4ThreeVector tangent(pp.position.y(),  -pp.position.x(), 0. );  
		    particle->SetPolarization(tangent.unit()); 
        }
        else if(m_torch->isPoint())
        {
            G4ThreeVector tangent(pp.position.y(),  -pp.position.x(), 0. );  
		    particle->SetPolarization(tangent.unit()); 
        }
        else if(m_torch->isSphere())
        {
            // see  cu/torchstep.h
            G4ThreeVector pol0(1,0,0); // <-- this direction needs to not be parallel to mom dir ?
            G4ThreeVector perp = pol0 - pol0.dot(pp.momentum_direction) * pp.momentum_direction;  
            // subtract vector component in the momentum direction, to yield perpendicular polarization
		    particle->SetPolarization(perp.unit()); 
        } 
        else
        {
		    particle->SetPolarization(m_polarization.x(), m_polarization.y(), m_polarization.z());
        }  


        if(m_torch->isFixPolarized())
        {
             //LOG(fatal) << "CTorchSource::GeneratePrimaryVertex" 
             //          << " fixpol override "
             //          << gformat(polarization)
             //          ; 
		     particle->SetPolarization(fixpol.unit());
        }


		if (m_verbosityLevel > 2) {
			LOG(info) << "Particle name: "
					  << m_definition->GetParticleName() 
                      << " verbosityLevel " << m_verbosityLevel
                      ;
			G4cout << "       Energy: " << pp.energy << G4endl;
			G4cout << "     Position: " << pp.position << G4endl;
			G4cout << "    Direction: " << pp.momentum_direction
					<< G4endl;
		}
		// Set bweight equal to the multiple of all non-zero weights
		G4double weight = m_eneGen->GetWeight()*m_ranGen->GetBiasWeight();
		particle->SetWeight(weight);

		vertex->SetPrimary(particle);
        evt->AddPrimaryVertex(vertex);

        collectPrimaryVertex(vertex);

	}
}





