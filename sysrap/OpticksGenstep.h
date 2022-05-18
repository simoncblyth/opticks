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
    OpticksGenstep_CARRIER                  = 14,
    OpticksGenstep_CERENKOV                 = 15,
    OpticksGenstep_SCINTILLATION            = 16,
    OpticksGenstep_NumType                  = 17
};
    

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#include <cstring>
#include "OpticksPhoton.h"

struct OpticksGenstep_
{
    static constexpr const char* INVALID_                 = "INVALID" ;
    static constexpr const char* G4Cerenkov_1042_         = "G4Cerenkov_1042" ;
    static constexpr const char* G4Scintillation_1042_    = "G4Scintillation_1042" ;
    static constexpr const char* DsG4Cerenkov_r3971_      = "DsG4Cerenkov_r3971" ;
    static constexpr const char* DsG4Scintillation_r3971_ = "DsG4Scintillation_r3971" ;
    static constexpr const char* DsG4Scintillation_r4695_ = "DsG4Scintillation_r4695" ;
    static constexpr const char* TORCH_                   = "TORCH" ;
    static constexpr const char* FABRICATED_              = "FABRICATED" ;
    static constexpr const char* EMITSOURCE_              = "EMITSOURCE" ; 
    static constexpr const char* NATURAL_                 = "NATURAL" ;
    static constexpr const char* MACHINERY_               = "MACHINERY" ;
    static constexpr const char* G4GUN_                   = "G4GUN" ;
    static constexpr const char* PRIMARYSOURCE_           = "PRIMARYSOURCE" ;
    static constexpr const char* GENSTEPSOURCE_           = "GENSTEPSOURCE" ;
    static constexpr const char* CARRIER_                 = "CARRIER" ;
    static constexpr const char* CERENKOV_                = "CERENKOV" ;
    static constexpr const char* SCINTILLATION_           = "SCINTILLATION" ;

    static unsigned Type(const char* name); 
    static const char* Name(unsigned type); 

    static bool IsValid(int gentype);
    static bool IsCerenkov(int gentype);
    static bool IsScintillation(int gentype);
    static bool IsTorchLike(int gentype);
    static bool IsEmitSource(int gentype);
    static bool IsMachinery(int gentype);
    static unsigned GenstepToPhotonFlag(int gentype);   
    static unsigned GentypeToPhotonFlag(char gentype); // 'C' 'S' 'T' -> CK, SI, TO

}; 

inline unsigned OpticksGenstep_::Type(const char* name) 
{
    unsigned type = OpticksGenstep_INVALID  ;
    if(strcmp(name,G4Cerenkov_1042_ )==0)         type = OpticksGenstep_G4Cerenkov_1042 ; 
    if(strcmp(name,G4Scintillation_1042_ )==0)    type = OpticksGenstep_G4Scintillation_1042 ; 
    if(strcmp(name,DsG4Cerenkov_r3971_ )==0)      type = OpticksGenstep_DsG4Cerenkov_r3971 ; 
    if(strcmp(name,DsG4Scintillation_r3971_ )==0) type = OpticksGenstep_DsG4Scintillation_r3971 ; 
    if(strcmp(name,DsG4Scintillation_r4695_ )==0) type = OpticksGenstep_DsG4Scintillation_r4695 ; 
    if(strcmp(name,TORCH_)==0)                    type = OpticksGenstep_TORCH ;
    if(strcmp(name,FABRICATED_)==0)               type = OpticksGenstep_FABRICATED ;
    if(strcmp(name,EMITSOURCE_)==0)               type = OpticksGenstep_EMITSOURCE ;
    if(strcmp(name,NATURAL_)==0)                  type = OpticksGenstep_NATURAL ;
    if(strcmp(name,MACHINERY_)==0)                type = OpticksGenstep_MACHINERY ;
    if(strcmp(name,G4GUN_)==0)                    type = OpticksGenstep_G4GUN ;
    if(strcmp(name,PRIMARYSOURCE_)==0)            type = OpticksGenstep_PRIMARYSOURCE ;
    if(strcmp(name,GENSTEPSOURCE_)==0)            type = OpticksGenstep_GENSTEPSOURCE ;
    if(strcmp(name,CARRIER_)==0)                  type = OpticksGenstep_CARRIER ;
    if(strcmp(name,CERENKOV_)==0)                 type = OpticksGenstep_CERENKOV ;
    if(strcmp(name,SCINTILLATION_)==0)            type = OpticksGenstep_SCINTILLATION ;
    return type ; 
}

inline const char* OpticksGenstep_::Name(unsigned type)
{
    const char* n = INVALID_ ; 
    switch(type)
    {   
        case OpticksGenstep_INVALID:                 n = INVALID_                 ; break ;  
        case OpticksGenstep_G4Cerenkov_1042:         n = G4Cerenkov_1042_         ; break ; 
        case OpticksGenstep_G4Scintillation_1042:    n = G4Scintillation_1042_    ; break ; 
        case OpticksGenstep_DsG4Cerenkov_r3971:      n = DsG4Cerenkov_r3971_      ; break ; 
        case OpticksGenstep_DsG4Scintillation_r3971: n = DsG4Scintillation_r3971_ ; break ; 
        case OpticksGenstep_DsG4Scintillation_r4695: n = DsG4Scintillation_r4695_ ; break ; 
        case OpticksGenstep_TORCH:                   n = TORCH_                   ; break ; 
        case OpticksGenstep_FABRICATED:              n = FABRICATED_              ; break ; 
        case OpticksGenstep_EMITSOURCE:              n = EMITSOURCE_              ; break ; 
        case OpticksGenstep_NATURAL:                 n = NATURAL_                 ; break ; 
        case OpticksGenstep_MACHINERY:               n = MACHINERY_               ; break ; 
        case OpticksGenstep_G4GUN:                   n = G4GUN_                   ; break ; 
        case OpticksGenstep_PRIMARYSOURCE:           n = PRIMARYSOURCE_           ; break ; 
        case OpticksGenstep_GENSTEPSOURCE:           n = GENSTEPSOURCE_           ; break ; 
        case OpticksGenstep_CARRIER:                 n = CARRIER_                 ; break ; 
        case OpticksGenstep_CERENKOV:                n = CERENKOV_                ; break ; 
        case OpticksGenstep_SCINTILLATION:           n = SCINTILLATION_           ; break ; 
        case OpticksGenstep_NumType:                 n = INVALID_                 ; break ; 
        default:                                     n = INVALID_                 ; break ; 
    }   
    return n ; 
}


inline bool OpticksGenstep_::IsValid(int gentype)   // static 
{  
   const char* s = Name(gentype);  
   bool invalid = strcmp(s, INVALID_) == 0 ;
   return !invalid ;
}

inline bool OpticksGenstep_::IsCerenkov(int gentype)  // static
{
   return gentype == OpticksGenstep_G4Cerenkov_1042  || 
          gentype == OpticksGenstep_DsG4Cerenkov_r3971 || 
          gentype == OpticksGenstep_CERENKOV
          ;
}
inline bool OpticksGenstep_::IsScintillation(int gentype)  // static
{
   return gentype == OpticksGenstep_G4Scintillation_1042 || 
          gentype == OpticksGenstep_DsG4Scintillation_r3971 || 
          gentype == OpticksGenstep_DsG4Scintillation_r4695 ||
          gentype == OpticksGenstep_SCINTILLATION 
         ;
}
inline bool OpticksGenstep_::IsTorchLike(int gentype)   // static
{
   return gentype == OpticksGenstep_TORCH || 
          gentype == OpticksGenstep_FABRICATED || 
          gentype == OpticksGenstep_EMITSOURCE 
          ;
} 
inline bool OpticksGenstep_::IsEmitSource(int gentype)   // static
{
   return gentype == OpticksGenstep_EMITSOURCE ;
} 
inline bool OpticksGenstep_::IsMachinery(int gentype)  // static
{
   return gentype == OpticksGenstep_MACHINERY ;
}

inline unsigned OpticksGenstep_::GenstepToPhotonFlag(int gentype)  // static
{
    unsigned phcode = 0 ;
    if(!IsValid(gentype))
    {
        phcode = NAN_ABORT ;
    }
    else if(IsCerenkov(gentype))
    {
        phcode = CERENKOV ;
    }
    else if(IsScintillation(gentype))
    {
        phcode = SCINTILLATION ;
    }
    else if(IsTorchLike(gentype))
    {
        phcode = TORCH ;
    }
    else
    {
        phcode = NAN_ABORT ;
    }
    return phcode ;
}

inline unsigned OpticksGenstep_::GentypeToPhotonFlag(char gentype)  // static
{
    unsigned phcode = 0 ;
    switch(gentype)
    {
        case 'C': phcode = CERENKOV          ; break ;
        case 'S': phcode = SCINTILLATION     ; break ;
        case 'T': phcode = TORCH             ; break ;
        default:  phcode = NAN_ABORT         ; break ;
    }
    return phcode ;
}


#endif


  
