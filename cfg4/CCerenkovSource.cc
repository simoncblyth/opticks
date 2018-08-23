#include "CFG4_BODY.hh"
#include <cassert>

#include "NGLM.hpp"
#include "NPri.hpp"

// g4-
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "G4PrimaryParticle.hh"
#include "G4Event.hh"
#include "G4ParticleMomentum.hh"
#include "G4ParticleTable.hh"
#include "G4ThreeVector.hh"

// cfg4-
#include "CCerenkovSource.hh"

#include "PLOG.hh"


unsigned CCerenkovSource::getNumG4Event() const { return m_pri->getNumG4Event() ; } 

CCerenkovSource::CCerenkovSource(Opticks* ok, NPY<float>* input_primaries, int verbosity)  
    :
    CSource(ok, verbosity),
    m_pri(new NPri(input_primaries)),
    m_event_count(0)
{
    init();
}

void CCerenkovSource::init()
{
}

CCerenkovSource::~CCerenkovSource() 
{
}


void CCerenkovSource::GeneratePrimaryVertex(G4Event *evt) 
{
    LOG(fatal) << "CCerenkovSource::GeneratePrimaryVertex" ;

    std::vector<unsigned> vtx_begin ; 
    std::vector<unsigned> vtx_count ; 
    findVertices(vtx_begin, vtx_count);
    assert( vtx_begin.size() == vtx_count.size() ); 

    unsigned num_vtx = vtx_begin.size() ; 
 
    for(unsigned v=0 ; v < num_vtx ; v++)
    {
        unsigned begin = vtx_begin[v] ;
        unsigned count = vtx_count[v] ;    

        G4PrimaryVertex* vtx = makePrimaryVertex(begin); 

        for( unsigned p=0 ; p < count  ; p++)
        {
            unsigned idx = begin + p ; 

            int eventIdx = m_pri->getEventIndex(idx); 
            int vertexIdx = m_pri->getVertexIndex(idx); 
            int particleIdx = m_pri->getParticleIndex(idx); 

            assert( unsigned(eventIdx) == m_event_count ) ; 
            assert( unsigned(vertexIdx) == v ); 
            assert( unsigned(particleIdx) == p ); 

            G4PrimaryParticle* pp = makePrimaryParticle(idx); 
            vtx->SetPrimary(pp) ; // NOTE poor API, actually adding not setting 
        }
        evt->AddPrimaryVertex( vtx );  
    }
    m_event_count += 1 ; 
}



   
/**
CCerenkovSource::makePrimaryParticle
------------------------------------

Make primary particle from the persisted record of one.

NB contrast with the converse in CPrimaryCollector::collectPrimaryParticle

**/


G4PrimaryVertex* CCerenkovSource::makePrimaryVertex(unsigned idx) const 
{
    glm::vec4 post = m_pri->getPositionTime(idx); 
    G4PrimaryVertex* vtx = new G4PrimaryVertex( post.x, post.y, post.z, post.w ); 
    return vtx ; 
}

G4PrimaryParticle* CCerenkovSource::makePrimaryParticle(unsigned idx) const 
{
    G4int pdgcode = m_pri->getPDGCode(idx) ; 

    G4ParticleTable* tab = G4ParticleTable::GetParticleTable();
    G4ParticleDefinition* pd = tab->FindParticle(pdgcode); 
    assert( pd ) ; 

    G4PrimaryParticle* pp = new G4PrimaryParticle( pd ); 

    glm::vec4 dirw = m_pri->getDirectionWeight(idx); 
    glm::vec4 polk = m_pri->getPolarizationKineticEnergy(idx); 

    G4ThreeVector dir(dirw.x, dirw.y, dirw.z); 
    pp->SetMomentumDirection(dir); 
    pp->SetWeight( dirw.w ); 

    pp->SetPolarization( polk.x, polk.y, polk.z ); 
    pp->SetKineticEnergy( polk.w ); 

    return pp ; 
}






