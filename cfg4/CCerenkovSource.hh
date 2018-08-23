#pragma once

#include "CSource.hh"
#include "CFG4_API_EXPORT.hh"
#include <vector>

class G4Event ; 
class G4PrimaryVertex ;
class G4PrimaryParticle ;
class G4VParticleChange ; 

class Opticks ; 
class NGS ; 
template <typename T> class NPY ; 

/**
CCerenkovSource
================

**/

class CFG4_API CCerenkovSource: public CSource
{
    public:
        CCerenkovSource(Opticks* ok,  NPY<float>* gs );
        virtual ~CCerenkovSource();
    private:
        void init();
    public:
        G4VParticleChange* generatePhotonsFromGenstep( unsigned i );
    public:
        // G4VPrimaryGenerator interface
        void GeneratePrimaryVertex(G4Event *evt);
    private:
        NGS*                  m_gs ;
        unsigned              m_generate_count ;   
        // event count should be in base class : but base needs a rewrite so leave it here for now


};


