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
#include "CPrimarySource.hh"

#include "PLOG.hh"


unsigned CPrimarySource::getNumG4Event() const { return m_pri->getNumG4Event() ; } 

CPrimarySource::CPrimarySource(Opticks* ok, NPY<float>* input_primaries, int verbosity)  
    :
    CSource(ok),
    m_pri(new NPri(input_primaries)),
    m_event_count(0)
{
    init();
}

void CPrimarySource::init()
{
}

CPrimarySource::~CPrimarySource() 
{
}


void CPrimarySource::GeneratePrimaryVertex(G4Event *evt) 
{
    LOG(fatal) << "CPrimarySource::GeneratePrimaryVertex" ;

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
CPrimarySource::findVertices
------------------------------

Each record in m_pri has indices for event/vertex/particle
allowing the correspondence between particle and vertex to be 
honoured.  Note the contiguous and ordered assumptions.

**/

void CPrimarySource::findVertices( std::vector<unsigned>& vtx_start, std::vector<unsigned>& vtx_count )
{
    unsigned num_pri = m_pri->getNumPri(); 

    for( unsigned idx=0 ; idx < num_pri ; idx++)
    {
        int eventIdx = m_pri->getEventIndex(idx); 
        int vertexIdx = m_pri->getVertexIndex(idx); 
        int particleIdx = m_pri->getParticleIndex(idx); 

        if( unsigned(eventIdx) == m_event_count && 
            unsigned(vertexIdx) == vtx_start.size() && 
            unsigned(particleIdx) == 0u) 
        {
            unsigned offset = 1 ;  // lookahead within the same vertex, until reach next vertex or end of pri  
            while( idx + offset < num_pri && m_pri->getVertexIndex(idx+offset) == vertexIdx )
            {
                assert( unsigned(m_pri->getParticleIndex(idx+offset)) == offset ); 
                offset += 1 ; 
            } 
            vtx_start.push_back(idx); 
            vtx_count.push_back(offset); 
        }
    }
}  


   
/**
CPrimarySource::makePrimaryParticle
------------------------------------

Make primary particle from the persisted record of one.

NB contrast with the converse in CPrimaryCollector::collectPrimaryParticle


**/


G4PrimaryVertex* CPrimarySource::makePrimaryVertex(unsigned idx) const 
{
    glm::vec4 post = m_pri->getPositionTime(idx); 
    G4PrimaryVertex* vtx = new G4PrimaryVertex( post.x, post.y, post.z, post.w ); 
    return vtx ; 
}

G4PrimaryParticle* CPrimarySource::makePrimaryParticle(unsigned idx) const 
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


/*
    G4ThreeVector position = GetParticlePosition();
    G4double time = GetParticleTime() ; 

    G4PrimaryVertex* vertex = new G4PrimaryVertex(position, time);

    G4double energy = GetParticleEnergy();

    G4ParticleMomentum direction = GetParticleMomentumDirection();
    G4ThreeVector polarization = GetParticlePolarization();

    G4double mass = m_definition->GetPDGMass() ;
    G4double charge = m_definition->GetPDGCharge() ;

    G4PrimaryParticle* primary = new G4PrimaryParticle(m_definition);

    primary->SetKineticEnergy( energy);
    primary->SetMass( mass );
    primary->SetMomentumDirection( direction);
    primary->SetCharge( charge );
	primary->SetPolarization(polarization.x(), polarization.y(), polarization.z()); 

    vertex->SetPrimary(primary);
    evt->AddPrimaryVertex(vertex);

 
    LOG(info) << "CPrimarySource::GeneratePrimaryVertex" 
              << " time " << time 
              << " position.x " << position.x() 
              << " position.y " << position.y() 
              << " position.z " << position.z() 
              ; 

*/









