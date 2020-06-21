#pragma once

/**
OpticksGenstep.h
=================

Genstep versioning

Not using typedef enum for simplicity as
this needs to be used everywhere. 

**/

enum
{
    OpticksGenstep_Invalid                  = 0, 
    OpticksGenstep_G4Cerenkov_1042          = 1,    
    OpticksGenstep_G4Scintillation_1042     = 2,    
    OpticksGenstep_DsG4Cerenkov_r3971       = 3,    
    OpticksGenstep_DsG4Scintillation_r3971  = 4, 
    OpticksGenstep_NumType                  = 5 
};
    
  
