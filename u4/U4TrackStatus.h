#pragma once
/**
U4TrackStatus.h
=================


  fAlive,             // Continue the tracking
  fStopButAlive,      // Invoke active rest physics processes and
                      // and kill the current track afterward
  fStopAndKill,       // Kill the current track

  fKillTrackAndSecondaries,
                      // Kill the current track and also associated
                      // secondaries.
  fSuspend,           // Suspend the current track
  fPostponeToNextEvent
                      // Postpones the tracking of thecurrent track 
                      // to the next event.

**/

#include "G4TrackStatus.hh"

struct U4TrackStatus
{
    static const char* Name(unsigned status); 
    static constexpr const char* fAlive_ = "fAlive" ; 
    static constexpr const char* fStopButAlive_ = "fStopButAlive" ; 
    static constexpr const char* fStopAndKill_ = "fStopAndKill" ; 
    static constexpr const char* fKillTrackAndSecondaries_ = "fKillTrackAndSecondaries" ; 
    static constexpr const char* fSuspend_ = "fSuspend" ; 
    static constexpr const char* fPostponeToNextEvent_ = "fPostponeToNextEvent" ; 
};

inline const char* U4TrackStatus::Name(unsigned status)
{
    const char* s = nullptr ; 
    switch(status)
    {
       case fAlive:                   s = fAlive_                   ; break ;  
       case fStopButAlive:            s = fStopButAlive_            ; break ;  
       case fStopAndKill:             s = fStopAndKill_             ; break ; 
       case fKillTrackAndSecondaries: s = fKillTrackAndSecondaries_ ; break ; 
       case fSuspend:                 s = fSuspend_                 ; break ; 
       case fPostponeToNextEvent:     s = fPostponeToNextEvent_     ; break ; 
    }
    return s ; 
}

