#pragma once

/**
OpticksGenstep.h
=================

Genstep versioning

Not using typedef enum for simplicity as
this needs to be used everywhere. 

NB these were formely conflated with photon flags, 
but the needs are somewhat different.

See also: npy/G4StepNPY.cpp  (TODO: consolidate these?)

**/

enum
{
    OpticksGenstep_INVALID                  = 0, 
    OpticksGenstep_G4Cerenkov_1042          = 1,    
    OpticksGenstep_G4Scintillation_1042     = 2,    
    OpticksGenstep_DsG4Cerenkov_r3971       = 3,    
    OpticksGenstep_DsG4Scintillation_r3971  = 4, 
    OpticksGenstep_DsG4Scintillation_r4695  = 5, 
    OpticksGenstep_TORCH                    = 6, 
    OpticksGenstep_FABRICATED               = 7, 
    OpticksGenstep_EMITSOURCE               = 8, 
    OpticksGenstep_NATURAL                  = 9, 
    OpticksGenstep_MACHINERY                = 10, 
    OpticksGenstep_G4GUN                    = 11, 
    OpticksGenstep_PRIMARYSOURCE            = 12, 
    OpticksGenstep_GENSTEPSOURCE            = 13, 
    OpticksGenstep_PHOTON_CARRIER           = 14,
    OpticksGenstep_CERENKOV                 = 15,
    OpticksGenstep_SCINTILLATION            = 16,
    OpticksGenstep_NumType                  = 17
};
    
  
