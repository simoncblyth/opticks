
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "CParticleDefinition.hh"

#include "PLOG.hh"


G4ParticleDefinition* CParticleDefinition::Find(const char* name) // static 
{ 
    G4ParticleTable* table = G4ParticleTable::GetParticleTable() ;
	G4ParticleDefinition* definition = table->FindParticle(name);
    bool known_particle = definition != NULL ; 
    if(!known_particle) 
    {
        LOG(fatal) << "CSource::FindParticle no particle with name [" << name << "] valid names listed below " ; 
        for(int i=0 ; i < table->entries() ; i++)
        {
             LOG(info) << std::setw(5) << i << " name [" << table->GetParticleName(i) << "]" ;  
        }
    } 
    assert(known_particle);
    return definition ;
}


