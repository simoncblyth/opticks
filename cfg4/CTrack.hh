#pragma once

// Simple G4Track wrapper providing getTrackStatusString()

class G4Track ;
#include "G4TrackStatus.hh"  
#include "CFG4_API_EXPORT.hh"
class CFG4_API CTrack {
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



