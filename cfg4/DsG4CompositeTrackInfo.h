#pragma once
// /usr/local/env/dyb/NuWa-trunk/dybgaudi/Simulation/G4DataHelpers/G4DataHelpers/G4CompositeTrackInfo.h

#include "G4VUserTrackInformation.hh"

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class CFG4_API DsG4CompositeTrackInfo : public G4VUserTrackInformation {
public:
     DsG4CompositeTrackInfo() : fHistoryTrackInfo(0), fPhotonTrackInfo(0) {} ;
     virtual ~DsG4CompositeTrackInfo();

     void SetHistoryTrackInfo(G4VUserTrackInformation* ti) { fHistoryTrackInfo=ti; }
     G4VUserTrackInformation* GetHistoryTrackInfo() { return fHistoryTrackInfo; }

     void SetPhotonTrackInfo(G4VUserTrackInformation* ti) { fPhotonTrackInfo=ti; }
     G4VUserTrackInformation* GetPhotonTrackInfo() { return fPhotonTrackInfo; }
     
     void Print() const {};
private:
     G4VUserTrackInformation* fHistoryTrackInfo;
     G4VUserTrackInformation* fPhotonTrackInfo;
};

#include "CFG4_TAIL.hh"



