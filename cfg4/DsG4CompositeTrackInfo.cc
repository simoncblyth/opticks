#include "DsG4CompositeTrackInfo.h"

DsG4CompositeTrackInfo::~DsG4CompositeTrackInfo()
{
    if (fHistoryTrackInfo)   delete fHistoryTrackInfo;
    if ( fPhotonTrackInfo )  delete fPhotonTrackInfo;
}

