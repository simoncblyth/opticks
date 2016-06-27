#pragma once

#include <vector>
#include <set>

#include "globals.hh"
#include "G4VUserPhysicsList.hh"

class OpNovicePhysicsListMessenger;

class G4Cerenkov;
//class G4Scintillation;
class Scintillation;
class G4OpAbsorption;

//class G4OpRayleigh;
class OpRayleigh;

class G4OpMieHG;
class G4OpBoundaryProcess;

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class CFG4_API OpNovicePhysicsList : public G4VUserPhysicsList
{
  public:

    OpNovicePhysicsList();
    virtual ~OpNovicePhysicsList();
  public:
    void Summary(const char* msg="OpNovicePhysicsList::Summary");
    void collectProcesses();
    void setProcessVerbosity(unsigned int verbosity);
    void setupEmVerbosity(unsigned int verbosity);
  public:
    void dump(const char* msg="OpNovicePhysicsList::dump");
  private:
    void dumpRayleigh(const char* msg="OpNovicePhysicsList::dumpRayleigh");
    void dumpMaterials(const char* msg="OpNovicePhysicsList::dumpMaterials");
    void dumpProcesses(const char* msg="OpNovicePhysicsList::dumpProcesses");
  public:

    virtual void ConstructParticle();
    virtual void ConstructProcess();

    virtual void SetCuts();

    //these methods Construct physics processes and register them
    void ConstructDecay();
    void ConstructEM();
    void ConstructOp();

    //for the Messenger 
    void SetVerbose(G4int);
    void SetNbOfPhotonsCerenkov(G4int);
 
  private:
    std::set<G4VProcess*> m_procs ; 
    std::vector<G4VProcess*> m_procl ; 

    OpNovicePhysicsListMessenger* fMessenger;

    static G4ThreadLocal G4int fVerboseLevel;
    static G4ThreadLocal G4int fMaxNumPhotonStep;

    static G4ThreadLocal G4Cerenkov* fCerenkovProcess;
    //static G4ThreadLocal G4Scintillation* fScintillationProcess;
    static G4ThreadLocal Scintillation* fScintillationProcess;
    static G4ThreadLocal G4OpAbsorption* fAbsorptionProcess;

    //static G4ThreadLocal G4OpRayleigh* fRayleighScatteringProcess;
    static G4ThreadLocal OpRayleigh* fRayleighScatteringProcess;

    static G4ThreadLocal G4OpMieHG* fMieHGScatteringProcess;
    static G4ThreadLocal G4OpBoundaryProcess* fBoundaryProcess;
};

#include "CFG4_TAIL.hh"


