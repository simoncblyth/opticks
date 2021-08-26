#pragma once

#include "plog/Severity.h"
#include "GGEO_API_EXPORT.hh"

/**
GGeoGen
=========

**/

class GGeo ; 
class OpticksGenstep ; 

class GGEO_API GGeoGen 
{
        static const plog::Severity LEVEL ; 
    public:
        GGeoGen(const GGeo* ggeo); 
        const OpticksGenstep* createDefaultTorchStep(unsigned num_photons, int node_index, unsigned originTrackID) const ; 
    private:
        const GGeo* m_ggeo ; 
};


