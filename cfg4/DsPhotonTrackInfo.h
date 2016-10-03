#pragma once

#include "G4VUserTrackInformation.hh"

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class CFG4_API DsPhotonTrackInfo : public G4VUserTrackInformation
{
public:
    enum QEMode 
    { 
            kQENone, 
            kQEPreScale, 
            kQEWater 
    };

    DsPhotonTrackInfo(QEMode mode=DsPhotonTrackInfo::kQENone, double qe=1.) ;


    QEMode GetMode() { return fMode; }
    void   SetMode(QEMode m) { fMode=m; }

    double GetQE() { return fQE; }
    void   SetQE(double qe) { fQE=qe; }

    bool GetReemitted() { return fReemitted; }
    void SetReemitted( bool re=true ) { fReemitted=re; }
    
    void Print() const {};
private:
    QEMode fMode;
    double fQE;
    bool   fReemitted;
};

#include "CFG4_TAIL.hh"
