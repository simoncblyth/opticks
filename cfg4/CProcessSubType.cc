#include "G4EmProcessSubType.hh"

#include "G4TransportationProcessType.hh"
#include "CProcessSubType.hh"

const char* CProcessSubType::fCoulombScattering_                 = "fCoulombScattering"                ;
const char* CProcessSubType::fIonisation_                        = "fIonisation"                       ;
const char* CProcessSubType::fBremsstrahlung_                    = "fBremsstrahlung"                   ;
const char* CProcessSubType::fPairProdByCharged_                 = "fPairProdByCharged"                ;
const char* CProcessSubType::fAnnihilation_                      = "fAnnihilation"                     ;
const char* CProcessSubType::fAnnihilationToMuMu_                = "fAnnihilationToMuMu"               ;
const char* CProcessSubType::fAnnihilationToHadrons_             = "fAnnihilationToHadrons"            ;
const char* CProcessSubType::fNuclearStopping_                   = "fNuclearStopping"                  ;
const char* CProcessSubType::fMultipleScattering_                = "fMultipleScattering"               ;
const char* CProcessSubType::fRayleigh_                          = "fRayleigh"                         ;
const char* CProcessSubType::fPhotoElectricEffect_               = "fPhotoElectricEffect"              ;
const char* CProcessSubType::fComptonScattering_                 = "fComptonScattering"                ;
const char* CProcessSubType::fGammaConversion_                   = "fGammaConversion"                  ;
const char* CProcessSubType::fGammaConversionToMuMu_             = "fGammaConversionToMuMu"            ;
const char* CProcessSubType::fCerenkov_                          = "fCerenkov"                         ;
const char* CProcessSubType::fScintillation_                     = "fScintillation"                    ;
const char* CProcessSubType::fSynchrotronRadiation_              = "fSynchrotronRadiation"             ;
const char* CProcessSubType::fTransitionRadiation_               = "fTransitionRadiation"              ;

const char* CProcessSubType::TRANSPORTATION_                     = "TRANSPORTATION"    ; 
const char* CProcessSubType::COUPLED_TRANSPORTATION_             = "COUPLED_TRANSPORTATION"    ; 
const char* CProcessSubType::STEP_LIMITER_                       = "STEP_LIMITER"    ; 
const char* CProcessSubType::USER_SPECIAL_CUTS_                  = "USER_SPECIAL_CUTS" ; 
const char* CProcessSubType::NEUTRON_KILLER_                     = "NEUTRON_KILLER" ; 



const char* CProcessSubType::Name(unsigned subtype) 
{
    const char* s = nullptr ; 
    switch(subtype)
    {
        case fCoulombScattering                  : s = fCoulombScattering_                 ; break ;
        case fIonisation                         : s = fIonisation_                        ; break ;
        case fBremsstrahlung                     : s = fBremsstrahlung_                    ; break ;
        case fPairProdByCharged                  : s = fPairProdByCharged_                 ; break ;
        case fAnnihilation                       : s = fAnnihilation_                      ; break ;
        case fAnnihilationToMuMu                 : s = fAnnihilationToMuMu_                ; break ;
        case fAnnihilationToHadrons              : s = fAnnihilationToHadrons_             ; break ;
        case fNuclearStopping                    : s = fNuclearStopping_                   ; break ;
        case fMultipleScattering                 : s = fMultipleScattering_                ; break ;
        case fRayleigh                           : s = fRayleigh_                          ; break ;
        case fPhotoElectricEffect                : s = fPhotoElectricEffect_               ; break ;
        case fComptonScattering                  : s = fComptonScattering_                 ; break ;
        case fGammaConversion                    : s = fGammaConversion_                   ; break ;
        case fGammaConversionToMuMu              : s = fGammaConversionToMuMu_             ; break ;
        case fCerenkov                           : s = fCerenkov_                          ; break ;
        case fScintillation                      : s = fScintillation_                     ; break ;
        case fSynchrotronRadiation               : s = fSynchrotronRadiation_              ; break ;
        case fTransitionRadiation                : s = fTransitionRadiation_               ; break ;

        case TRANSPORTATION                      : s = TRANSPORTATION_                     ; break ; 
        case COUPLED_TRANSPORTATION              : s = COUPLED_TRANSPORTATION_             ; break ; 
        case STEP_LIMITER                        : s = STEP_LIMITER_                       ; break ; 
        case USER_SPECIAL_CUTS                   : s = USER_SPECIAL_CUTS_                  ; break ; 
        case NEUTRON_KILLER                      : s = NEUTRON_KILLER_                     ; break ;  
    }
    return s ; 
}

