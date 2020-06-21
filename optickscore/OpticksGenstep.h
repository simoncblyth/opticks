#pragma once

/**
OpticksGenstep.h
=================

Genstep versioning

Not using typedef enum for simplicity as
this needs to be used everywhere. 

NB these were formely conflated with photon flags, 
but the needs are somewhat different.


**/

enum
{
    OpticksGenstep_INVALID                  = 0, 
    OpticksGenstep_G4Cerenkov_1042          = 1,    
    OpticksGenstep_G4Scintillation_1042     = 2,    
    OpticksGenstep_DsG4Cerenkov_r3971       = 3,    
    OpticksGenstep_DsG4Scintillation_r3971  = 4, 
    OpticksGenstep_TORCH                    = 5, 
    OpticksGenstep_FABRICATED               = 6, 
    OpticksGenstep_EMITSOURCE               = 7, 
    OpticksGenstep_NATURAL                  = 8, 
    OpticksGenstep_MACHINERY                = 9, 
    OpticksGenstep_G4GUN                    = 10, 
    OpticksGenstep_PRIMARYSOURCE            = 11, 
    OpticksGenstep_GENSTEPSOURCE            = 12, 
    OpticksGenstep_NumType                  = 13 
};
    
  
