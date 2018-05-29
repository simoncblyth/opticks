#pragma once

#include <vector>
#include <map>

#include "G4OK_API_EXPORT.hh"

class OpMgr;
class G4Run;
class G4Event; 

class G4OK_API G4OpticksManager   
{
  public:
    static G4OpticksManager* GetOpticksManager();
  public:
    G4OpticksManager();
    virtual ~G4OpticksManager();

  public:

    virtual void BeginOfRunAction(const G4Run*);
    virtual void EndOfRunAction(const G4Run*);
    virtual void BeginOfEventAction(const G4Event*);
    virtual void EndOfEventAction(const G4Event*);

  public:
    // Optical photon producing processes such as Scintillation and Cerenkov
    // need to invoke `addGenstep` at every step.  
    // Thus gensteps collected ready for on GPU propagation at the end of the event, 
    // rather than generating the photons.

    void addGenstep( float* data, unsigned num_float );

  private:
    // invoked internally from BeginOfRun action 
    void checkGeometry();
    void checkMaterials();
    void setupPropagator();

  private:
    void propagate(int eventId);

  private:
    OpMgr*      m_opmgr;
    const char* m_lookup ; 
    std::map<std::string, int> m_mat_g; // geant4 mat name: index
    std::vector<int> m_g2c; // mapping of mat idx: geant4 to opticks

  private:
     //static G4ThreadLocal G4OpticksManager*  fOpticksManager;
     static G4OpticksManager*  fOpticksManager;


};



