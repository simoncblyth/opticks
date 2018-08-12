#pragma once

#include <string>
#include <vector>
#include <map>

#include "G4OK_API_EXPORT.hh"

class NLookup ; 
class Opticks;
class OpMgr;
class GGeo ; 
class GBndLib ; 

class CTraverser ; 
class CMaterialTable ; 
class CCollector ; 

class G4Run;
class G4Event; 
class G4VPhysicalVolume ;

#include "G4Types.hh"

/**
G4Opticks : interface for steering an Opticks instance embedded within a Geant4 application
==============================================================================================

setGeometry 
   called from BeginOfRunAction passing the world pointer over 
   in order to translate and serialize the geometry and copy it
   to the GPU  

addGenstep
   called from every step of modified Scintillation and Cerenkov processees
   in place of the optical photon generation loop, all the collected 
   gensteps are propagated together in a single GPU launch when propagate
   is called

propagate and getHits
   called from EndOfEventAction 


Notes
-------

* :doc:`notes/issues/G4OK`


**/

class G4OK_API G4Opticks   
{
    private:
        static const char* fEmbeddedCommandLine ; 
    public:
        static G4Opticks* GetOpticks();
    public:
        G4Opticks();
        virtual ~G4Opticks();
    public:
        std::string desc();  
    public:
        void setGeometry(const G4VPhysicalVolume* world); 
    private:
        GGeo* translateGeometry( const G4VPhysicalVolume* top );
        void setupMaterialLookup();
    public:
        void collectCerenkovStep(
            G4int                id, 
            G4int                parentId,
            G4int                materialId,
            G4int                numPhotons,
    
            G4double             x0_x,  
            G4double             x0_y,  
            G4double             x0_z,  
            G4double             t0, 

            G4double             deltaPosition_x, 
            G4double             deltaPosition_y, 
            G4double             deltaPosition_z, 
            G4double             stepLength, 

            G4int                pdgCode, 
            G4double             pdgCharge, 
            G4double             weight, 
            G4double             meanVelocity, 

            G4double             betaInverse,
            G4double             pmin,
            G4double             pmax,
            G4double             maxCos,

            G4double             maxSin2,
            G4double             meanNumberOfPhotons1,
            G4double             meanNumberOfPhotons2,
            G4double             spare2=0
        );  

    private:
        const G4VPhysicalVolume*   m_world ; 
        const GGeo*                m_ggeo ; 
        const GBndLib*             m_blib ; 
        Opticks*                   m_ok ;
        CTraverser*                m_traverser ; 
        CMaterialTable*            m_mtab ; 
        CCollector*                m_collector ; 
        NLookup*                   m_lookup ; 
        OpMgr*                     m_opmgr;
    private:
        static G4Opticks*          fOpticks;

};



