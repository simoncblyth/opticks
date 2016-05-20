#pragma once

enum
{
    CERENKOV          = 0x1 <<  0,    
    SCINTILLATION     = 0x1 <<  1,    
    MISS              = 0x1 <<  2,
    BULK_ABSORB       = 0x1 <<  3,
    BULK_REEMIT       = 0x1 <<  4,
    BULK_SCATTER      = 0x1 <<  5,
    SURFACE_DETECT    = 0x1 <<  6,
    SURFACE_ABSORB    = 0x1 <<  7,
    SURFACE_DREFLECT  = 0x1 <<  8,
    SURFACE_SREFLECT  = 0x1 <<  9,
    BOUNDARY_REFLECT  = 0x1 << 10,
    BOUNDARY_TRANSMIT = 0x1 << 11,
    TORCH             = 0x1 << 12,
    NAN_ABORT         = 0x1 << 13,
    G4GUN             = 0x1 << 14
}; 

//  only ffs 0-15 make it into the record so debug flags only beyond 15 




