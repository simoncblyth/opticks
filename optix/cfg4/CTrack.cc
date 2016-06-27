#include "CFG4_BODY.hh"
#include "CTrack.hh"
#include "G4Track.hh"

const char* CTrack::fAlive_                    = "fAlive" ;
const char* CTrack::fStopButAlive_             = "fStopButAlive" ;
const char* CTrack::fStopAndKill_              = "fStopAndKill" ;
const char* CTrack::fKillTrackAndSecondaries_  = "fKillTrackAndSecondaries" ;
const char* CTrack::fSuspend_                  = "fSuspend" ;
const char* CTrack::fPostponeToNextEvent_      = "fPostponeToNextEvent" ;

CTrack::CTrack(G4Track* track) 
   :
     m_track(track)
{
}

const char* CTrack::TrackStatusString(G4TrackStatus status)
{
   const char* s = NULL ; 
   switch(status)
   {  
      case fAlive:                     s=fAlive_                   ;break; // Continue the tracking
      case fStopButAlive:              s=fStopButAlive_            ;break; // Invoke active rest physics processes and and kill the current track afterward
      case fStopAndKill :              s=fStopAndKill_             ;break; // Kill the current track
      case fKillTrackAndSecondaries:   s=fKillTrackAndSecondaries_ ;break; // Kill the current track and also associated secondaries.
      case fSuspend:                   s=fSuspend_                 ;break; // Suspend the current track
      case fPostponeToNextEvent:       s=fPostponeToNextEvent_     ;break; // Postpones the tracking of thecurrent track to the next event.
  }
  return s ; 
} 

const char* CTrack::getTrackStatusString()
{
   return TrackStatusString(m_track->GetTrackStatus());
}


