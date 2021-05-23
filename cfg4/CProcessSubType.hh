#pragma once

/**
CProcessSubType
====================

See CProcessSubType.sh for generation with bin/enu.py 


To find unidentified subtypes use::

    BP=G4VProcess::SetProcessSubType ./run.sh 

**/

struct CProcessSubType
{
    static const char* fCoulombScattering_ ;
    static const char* fIonisation_ ;
    static const char* fBremsstrahlung_ ;
    static const char* fPairProdByCharged_ ;
    static const char* fAnnihilation_ ;
    static const char* fAnnihilationToMuMu_ ;
    static const char* fAnnihilationToHadrons_ ;
    static const char* fNuclearStopping_ ;
    static const char* fMultipleScattering_ ;
    static const char* fRayleigh_ ;
    static const char* fPhotoElectricEffect_ ;
    static const char* fComptonScattering_ ;
    static const char* fGammaConversion_ ;
    static const char* fGammaConversionToMuMu_ ;
    static const char* fCerenkov_ ;
    static const char* fScintillation_ ;
    static const char* fSynchrotronRadiation_ ;
    static const char* fTransitionRadiation_ ;

    static const char* TRANSPORTATION_ ; 
    static const char* COUPLED_TRANSPORTATION_ ; 
    static const char* STEP_LIMITER_ ; 
    static const char* USER_SPECIAL_CUTS_ ; 
    static const char* NEUTRON_KILLER_ ; 

    static const char* Name(unsigned subtype) ; 
}; 



