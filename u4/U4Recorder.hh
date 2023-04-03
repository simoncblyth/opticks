#pragma once
/**
U4Recorder
===========

This is used from tests/U4App.h (the Geant4 application in a header)

U4Recorder is NOT a G4UserRunAction, G4UserEventAction, 
... despite having the corresponding method names. 

The U4Recorder relies on the RunAction, EventAction  etc.. classes 
calling those lifecycle methods.   





**/

#include <vector>
#include <string>

class G4Run ; 
class G4Event ; 
class G4Track ; 
class G4Step ; 
class G4VSolid ; 
class G4StepPoint ; 

struct NP ; 
struct spho ; 
struct quad4 ; 
struct SEvt ; 

#include "NPU.hh"  // UName

#include "plog/Severity.h"
#include "G4TrackStatus.hh"
#include "U4_API_EXPORT.hh"

struct U4_API U4Recorder 
{
    static const plog::Severity LEVEL ; 
    static UName SPECS ;      // collect unique U4Step::Spec strings  

    static const int STATES ; // configures number of g4states to persist 
    static const int RERUN  ; 
    static constexpr int STATE_ITEMS = 2*17+4 ; // 2*17+4 is appropriate for MixMaxRng 

    static const bool PIDX_ENABLED ; 

    static constexpr const char* EndOfRunAction_Simtrace_ = "U4Recorder__EndOfRunAction_Simtrace" ; 
    static const bool            EndOfRunAction_Simtrace ; 

    static const int PIDX ;   // used to control debug printout for idx 
    static const int EIDX ;   // used to enable U4Recorder for an idx, skipping all others
    static const int GIDX ; 

    static std::string Switches(); 
    static std::string Desc(); 
    static bool Enabled(const spho& label); 

    static U4Recorder* INSTANCE ; 
    static U4Recorder* Get(); 

    U4Recorder(); 

    int eventID ; 
    const G4Track* transient_fSuspend_track ; 
    NP* rerun_rand ;  
    SEvt* evt ; 


    void BeginOfRunAction(const G4Run*);
    void EndOfRunAction(const G4Run*);

    void BeginOfEventAction(const G4Event*);
    void EndOfEventAction(const G4Event*);

    void PreUserTrackingAction(const G4Track*);
    void PostUserTrackingAction(const G4Track*);

    void PreUserTrackingAction_Optical(const G4Track*);

    void PreUserTrackingAction_Optical_GetLabel( spho& ulabel, const G4Track* track ); 
    void PreUserTrackingAction_Optical_FabricateLabel( const G4Track* track ) ; 
    void GetLabel( spho& ulabel, const G4Track* track ); 


    void saveOrLoadStates(int id); 
    void saveRerunRand(const char* dir) const ; 

    static NP* MakeMetaArray() ; 
    static void SaveMeta(const char* savedir); 

    void PostUserTrackingAction_Optical(const G4Track*);


    void UserSteppingAction(const G4Step*);

    // boundary process template type
    template<typename T> 
    void UserSteppingAction_Optical(const G4Step*); 




    template <typename T>
    static void CollectBoundaryAux(quad4* current_aux ); 


    static const double EPSILON ; 
    unsigned ClassifyFake(const G4Step* step, unsigned flag, const char* spec, bool dump); 

    static std::vector<std::string>* FAKES ;       // envvar U4Recorder__FAKES
    static bool                      FAKES_SKIP ;  // envvar U4Recorder__FAKES_SKIP 

    static bool IsListed( const std::vector<std::string>* LIST, const char* spec ) ;
    static bool IsListedFake( const char* spec ); 
    static std::string DescFakes(); 


    void Check_TrackStatus_Flag(G4TrackStatus tstat, unsigned flag, const char* from ); 
};

