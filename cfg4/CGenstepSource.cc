#include "CFG4_BODY.hh"
#include <cassert>

#include "NGLM.hpp"
#include "NGS.hpp"
#include "Opticks.hh"

// g4-
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "G4ParticleChange.hh"
#include "G4PrimaryParticle.hh"
#include "G4Event.hh"
#include "G4ParticleMomentum.hh"
#include "G4ParticleTable.hh"
#include "G4ThreeVector.hh"

// cfg4-
#include "CGenstepSource.hh"
#include "CGenstepSource.hh"

#include "PLOG.hh"


CGenstepSource::CGenstepSource(Opticks* ok, NPY<float>* gs)  
    :
    CSource(ok, ok->getVerbosity()),
    m_gs(new NGS(gs)),
    m_num_genstep(m_gs->getNumGensteps()),
    m_idx(0),
    m_generate_count(0)
{
    init();
}
void CGenstepSource::init()
{
}

CGenstepSource::~CGenstepSource() 
{
}


G4VParticleChange* CGenstepSource::generatePhotonsFromNextGenstep()
{
    assert( m_idx < m_num_genstep ); 
    G4VParticleChange* pc = CCerenkovGenerator::GeneratePhotonsFromGenstep(m_gs,m_idx) ;
    m_idx += 1 ; 
    return pc ; 
}


void CGenstepSource::GeneratePrimaryVertex(G4Event *evt) 
{
    LOG(fatal) << "CGenstepSource::GeneratePrimaryVertex" ;

    G4VParticleChange* pc = generatePhotonsFromNextGenstep() ; 


    m_generate_count += 1 ; 
}






