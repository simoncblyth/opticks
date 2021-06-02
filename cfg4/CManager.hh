#pragma once

#include "plog/Severity.h"

class G4Run ; 
class G4Event ; 
class G4Track ; 
class G4Step ; 
class G4Navigator ; 
class G4TransportationManager ; 

class Opticks ; 
class OpticksEvent ; 
class CRecorder ;
class CRandomEngine ; 
class CMaterialBridge ; 

class CMaterialLib ; 
class CRecorder ; 
class CStepRec ; 

struct CG4Ctx ; 


/**
CManager  (maybe CRouter is better name) 
=============================================

Middle management operating beneath CG4 and G4OpticksRecorder levels 
such that it can be used by both those.

So for example the manager will be what the geant4 actions talk to, 
rather than CG4 which is too high level for reusabliity. 


CManager accepts steps from Geant4, routing them to either:

1. m_recorder(CRecorder) for optical photon steps
2. m_noprec(CStepRec) for non-optical photon steps

The setStep method returns a boolean "done" which
dictates whether to fStopAndKill the track.
The mechanism is used to stop tracking when reach truncation (bouncemax)
as well as absorption.


dynamic vs static 
-------------------

When have the gensteps know exactly how many photons are
going to handle ahead of time so it is called static or non-dynamic running.   
With GPU simulation with Opticks always have all the gensteps ahead of time
and thus fixed size of photon buffers.

When doing Geant4 running with Opticks instrumentation it is more
common to run without knowing the gensteps ahead of time.

See CWriter::initEvent for the split between dynamic and static 


Whats the difference between CRecorder/m_recorder and CStepRec/m_noprec ?
-------------------------------------------------------------------------------

CStepRec 
   records non-optical particle steps into the m_nopstep buffer of the
   OpticksEvent set by CStepRec::initEvent
    
CRecorder
   records optical photon steps and photon tracks

CStepRec is beautifully simple, CRecorder is horribly complicated in comparison


**/

#include "CFG4_API_EXPORT.hh"

struct CFG4_API CManager
{
    static const plog::Severity LEVEL ; 
    static CManager* fINSTANCE ; 
    static CManager* Get(); 

    Opticks*          m_ok ; 
    bool              m_onestep ; 
    unsigned          m_mode ;     // --managermode 
    CG4Ctx*           m_ctx ; 
    CRandomEngine*    m_engine ; 
    CRecorder*        m_recorder   ; 
    CStepRec*         m_noprec   ; 

    bool              m_dbgflat ; 
    bool              m_dbgrec ; 
    G4TransportationManager* m_trman ; 
    G4Navigator*             m_nav ; 


    unsigned int m_noprec_store_count ;
    int          m_cursor_at_clear ;



    void setMaterialBridge(const CMaterialBridge* material_bridge);

    double             flat_instrumented(const char* file, int line);
    CRandomEngine*     getRandomEngine() const ; 
    CStepRec*          getStepRec() const ;
    CRecorder*         getRecorder() const ;
    CG4Ctx&            getCtx() ;
    unsigned long long getSeqHis() const ;
    void report(const char* msg="CManager::report");


    CManager(Opticks* ok, bool onestep); 


    // inputs from G4Opticks/G4OpticksRecorder

    void BeginOfGenstep(char gentype, int num_photons, int offset);
    void EndOfGenstep();


    // inputs from Geant4

    virtual void BeginOfRunAction(const G4Run*);
    virtual void EndOfRunAction(const G4Run*);

    virtual void BeginOfEventAction(const G4Event*);
    virtual void EndOfEventAction(const G4Event*);

    virtual void PreUserTrackingAction(const G4Track*);
    virtual void PostUserTrackingAction(const G4Track*);

    virtual void UserSteppingAction(const G4Step*);




    // actions
    void presave(); 
    void initEvent(OpticksEvent* evt);
    void save(); 




    void preTrack(); 
    void postTrack(); 

    bool setStep( const G4Step* step);
    void prepareForNextStep( const G4Step* step, G4Track* mtrack);
    void postStep(); 

    void postpropagate();
    void addRandomNote(const char* note, int value);
    void addRandomCut(const char* ckey, double cvalue);

};

