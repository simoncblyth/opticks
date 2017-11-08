#pragma once

#include "G4OpticalSurface.hh"
#include "CFG4_API_EXPORT.hh"

class CFG4_API COptical
{
    public:
       static G4OpticalSurfaceModel  Model(unsigned model_) ; 
       static G4OpticalSurfaceFinish Finish(unsigned finish_);
       static G4SurfaceType          Type(unsigned type_);
};


