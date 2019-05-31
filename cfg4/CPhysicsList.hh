#pragma once

#include <vector>
#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

/**
CPhysicsList
=============

* A reimplementation of the messy OpNovicePhysicsList 

TODO:

Follow the CCerenokov cleanup with analogous classes for Boundary and Scintillation


**/

class CG4 ; 
class Opticks ; 
class CCerenkov ; 
class G4ParticleDefinition ; 
class G4VProcess ; 

#include "G4VUserPhysicsList.hh"

class CFG4_API CPhysicsList : public G4VUserPhysicsList
{
    public:
        static const CPhysicsList* INSTANCE ; 
        CPhysicsList(CG4* g4);
        virtual ~CPhysicsList();
    public:
        // fulfil the interface
        void ConstructParticle();
        void ConstructProcess();
    public:
        void setupEmVerbosity(unsigned verbosity);
        void setProcessVerbosity(int verbosity);
        void setProcessVerbosity(G4ParticleDefinition* particle, int verbosity);
    private:
        void initParticles();
        void constructDecay();
        void constructEM();
        void constructOp();
        void constructEM(G4ParticleDefinition* particle);
        void constructOp(G4ParticleDefinition* particle);
    private:
        CG4*         m_g4 ; 
        Opticks*     m_ok ; 
        unsigned     m_emVerbosity ; 
        CCerenkov*   m_cerenkov ; 
        G4VProcess*  m_cerenkovProcess ; 
        G4VProcess*  m_scintillationProcess ; 
        G4VProcess*  m_boundaryProcess ; 
        G4VProcess*  m_absorptionProcess ; 
        G4VProcess*  m_rayleighProcess ; 

        typedef std::vector<G4ParticleDefinition*> VP ; 
        VP m_particles ; 
};

#include "CFG4_TAIL.hh"

