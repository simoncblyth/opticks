#pragma once

#include "CSource.hh"
#include "CFG4_API_EXPORT.hh"
#include <vector>

class G4Event ; 
class G4PrimaryVertex ;
class G4PrimaryParticle ;

template <typename T> class NPY ; 


/**
CCerenkovSource
================

**/

class CFG4_API CCerenkovSource: public CSource
{
    public:
        unsigned getNumG4Event() const ;
        CCerenkovSource(Opticks* ok,  NPY<float>* gs, int verbosity);
        virtual ~CCerenkovSource();
    private:
        void init();
        G4PrimaryVertex*   makePrimaryVertex(unsigned idx) const ;
        G4PrimaryParticle* makePrimaryParticle(unsigned idx) const ;
    public:
        // G4VPrimaryGenerator interface
        void GeneratePrimaryVertex(G4Event *evt);
    public:
    private:
        NPY<float>*           m_gs ;
        unsigned              m_event_count ;   
        // event count should be in base class : but base needs a rewrite so leave it here for now


};


