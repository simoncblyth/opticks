#pragma once

#include "X4_API_EXPORT.hh"

class G4OpticalSurface ; 
class GOpticalSurface ; 

/**
X4OpticalSurface
==================

CAUTION : Only a small fraction of Geant4 optical surface handling 
has been ported to Opticks.

**/

class X4_API X4OpticalSurface 
{
    public:
        static const char* Type(G4SurfaceType type);
        static GOpticalSurface* Convert(const G4OpticalSurface* const src );
};



