#pragma once

#include "G4OK_API_EXPORT.hh"

class G4OK_API G4OpticksScintillationStep {
    public:

    enum {

       _Id, 
       _ParentID,
       _Material,
       _NumPhotons,
      
       _x0_x,
       _x0_y,
       _x0_z,
       _t0,

       _DeltaPosition_x,
       _DeltaPosition_y,
       _DeltaPosition_z,
       _step_length,

       _code,
       _charge, 
       _weight, 
       _MeanVelocity,

       _scnt,  
       _slowerRatio,   
       _slowTimeConstant,    
       _slowerTimeConstant,

       _ScintillationTime,
       _ScintillationIntegralMax,
       _Spare1,
       _Spare2,

       SIZE

    };

};


