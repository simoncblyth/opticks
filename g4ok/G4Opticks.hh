#pragma once

#include <string>
#include <vector>
#include <map>

#include "G4OK_API_EXPORT.hh"


class Opticks;
class OpMgr;
class GGeo ; 
class CTraverser ; 
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


TODO
-----

1. bring in the X4 direct translation 
2. workout how to live boot the OpticksHub inside OpMgr with the directly translated geomtry 
3. automate material mapping ? 
  
   * in CCollector the NLookup translates raw G4 materialIdx into a GBndLib texture line 
     this is definitely needed : but it should be possible to entirely automate it 
     from the direct geometry and thus get it out of the interface

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
    // hmm : these are too generic : its better to inform the user whats happening 
    virtual void BeginOfRunAction(const G4Run*);
    virtual void EndOfRunAction(const G4Run*);
    virtual void BeginOfEventAction(const G4Event*);
    virtual void EndOfEventAction(const G4Event*);
  public:
    void setGeometry(const G4VPhysicalVolume* world); 
    void addGenstep( float* data, unsigned num_float=4*6 );

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
    // invoked internally from BeginOfRun action 
    void checkGeometry();
    void checkMaterials();
    void setupPropagator();

  private:
    GGeo* translateGeometry( const G4VPhysicalVolume* top );
    void propagate(int eventId);

  private:
    const G4VPhysicalVolume*   m_world ; 
    GGeo*                      m_ggeo ; 
    Opticks*                   m_ok ;
    OpMgr*                     m_opmgr;
    CTraverser*                m_traverser ; 
    CCollector*                m_collector ; 
    const char*                m_lookup ; 
    std::map<std::string, int> m_mat_g; // geant4 mat name: index
    std::vector<int>           m_g2c; // mapping of mat idx: geant4 to opticks

  private:
     static G4Opticks*  fOpticks;


};



