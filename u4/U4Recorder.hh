#pragma once
/**
U4Recorder
===========

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

#include "plog/Severity.h"
#include "G4TrackStatus.hh"
#include "U4_API_EXPORT.hh"

struct U4_API U4Recorder 
{
    static const plog::Severity LEVEL ; 

    static const int STATES ; // configures number of g4states to persist 
    static const int RERUN  ; 
    static constexpr int STATE_ITEMS = 2*17+4 ; // 2*17+4 is appropriate for MixMaxRng 

    static const int PIDX ;   // used to control debug printout for idx 
    static const int EIDX ;   // used to enable U4Recorder for an idx, skipping all others
    static const int GIDX ; 
    static std::string Desc(); 
    static bool Enabled(const spho& label); 

    static U4Recorder* INSTANCE ; 
    static U4Recorder* Get(); 

    U4Recorder(); 
    // NO MEMBERS : persisting is handled at lower level by sysrap/SEvt 

    void BeginOfRunAction(const G4Run*);
    void EndOfRunAction(const G4Run*);

    void BeginOfEventAction(const G4Event*);
    void EndOfEventAction(const G4Event*);

    void PreUserTrackingAction(const G4Track*);
    void PostUserTrackingAction(const G4Track*);

    void PreUserTrackingAction_Optical(const G4Track*);
    void saveOrLoadStates(int id); 

    void PostUserTrackingAction_Optical(const G4Track*);

    template<typename T> void UserSteppingAction(const G4Step*);
    template<typename T> void UserSteppingAction_Optical(const G4Step*); 

    void Check_TrackStatus_Flag(G4TrackStatus tstat, unsigned flag, const char* from ); 
};

