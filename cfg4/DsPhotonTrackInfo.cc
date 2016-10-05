#include "DsPhotonTrackInfo.h"

DsPhotonTrackInfo::DsPhotonTrackInfo( QEMode mode, double qe )
    : 
    fMode(mode), 
    fQE(qe), 
    fReemitted(false),
    fPrimaryPhotonID(-1)
{
}


