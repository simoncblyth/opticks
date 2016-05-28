#pragma once

class G4Track ;
#include "G4TrackStatus.hh"  

class CTrack {
   public:
    static const char* fAlive_ ;
    static const char* fStopButAlive_ ;
    static const char* fStopAndKill_ ;
    static const char* fKillTrackAndSecondaries_ ;
    static const char* fSuspend_ ;
    static const char* fPostponeToNextEvent_ ;
   public:
      CTrack(G4Track* track);
      const char* getTrackStatusString();
      static const char* TrackStatusString(G4TrackStatus status);
   private:
      G4Track* m_track ; 
};


inline CTrack::CTrack(G4Track* track) 
   :
     m_track(track)
{
}


