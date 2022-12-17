#pragma once
/**
CustomART
===========

Trying to do less than CustomBoundary, instead just calculate::

    theTransmittance
    theReflectivity
    theEfficiency 

TODO: should also probably be setting::
   
   type = dielectric_dielectric 
   theFinish = polished 

With everything else (deciding on ARTD, changing mom, pol, theStatus)
done by the nearly "standard" G4OpBoundaryProcess.   

**/

#include "G4ThreeVector.hh"

#include "SLOG.hh"
#include "JPMT.h"
#include "Layr.h"   // hmm use stmm.h  ? 
#include "U4Touchable.h"
#include "CustomStatus.h"


template<typename J>
struct CustomART
{
    static const constexpr plog::Severity LEVEL = debug ;  
    J*     parameter_accessor ; 
    int    count ; 
    double zlocal ; 
    double lposcost ; 

    G4double& theTransmittance ;
    G4double& theReflectivity ;
    G4double& theEfficiency ;

    const G4ThreeVector& theGlobalPoint ; 
    const G4ThreeVector& OldMomentum ; 
    const G4ThreeVector& OldPolarization ; 
    const G4ThreeVector& theRecoveredNormal ; 
    const G4double& thePhotonMomentum ; 

    CustomART(
        G4double& theTransmittance,
        G4double& theReflectivity,
        G4double& theEfficiency,
        const G4ThreeVector& theGlobalPoint,  
        const G4ThreeVector& OldMomentum,  
        const G4ThreeVector& OldPolarization,
        const G4ThreeVector& theRecoveredNormal,
        const G4double& thePhotonMomentum
    );  

    char maybe_doIt(const char* OpticalSurfaceName, const G4Track& aTrack, const G4Step& aStep) ;  
    char doIt(const G4Track& aTrack, const G4Step& aStep ); 
}; 

template<typename J>
inline CustomART<J>::CustomART(
          G4double& theTransmittance_,
          G4double& theReflectivity_,
          G4double& theEfficiency_,
    const G4ThreeVector& theGlobalPoint_,
    const G4ThreeVector& OldMomentum_,
    const G4ThreeVector& OldPolarization_,
    const G4ThreeVector& theRecoveredNormal_,
    const G4double&      thePhotonMomentum_
    )
    :
    parameter_accessor(new J),
    count(0),
    zlocal(-1.),
    lposcost(-2.),
    theTransmittance(theTransmittance_),
    theReflectivity(theReflectivity_),
    theEfficiency(theEfficiency_),
    theGlobalPoint(theGlobalPoint_),
    OldMomentum(OldMomentum_),
    OldPolarization(OldPolarization_),
    theRecoveredNormal(theRecoveredNormal_),
    thePhotonMomentum(thePhotonMomentum_) 
{
}

template<typename J>
inline char CustomART<J>::maybe_doIt(const char* OpticalSurfaceName, const G4Track& aTrack, const G4Step& aStep )
{
    if( OpticalSurfaceName == nullptr || OpticalSurfaceName[0] != '@') return 'N' ; 

    const G4AffineTransform& transform = aTrack.GetTouchable()->GetHistory()->GetTopTransform();
    G4ThreeVector localPoint = transform.TransformPoint(theGlobalPoint);
    zlocal = localPoint.z() ; 
    lposcost = localPoint.cosTheta() ;  
    // Q:What is lposcost for ?  
    // A:Preparing for doing this on GPU, as lposcost is available there already but not zlocal 

    if(zlocal <= 0) return 'Z' ; 

    return doIt(aTrack, aStep) ;
}

template<typename J>
inline char CustomART<J>::doIt(const G4Track& aTrack, const G4Step& aStep )
{
    G4double minus_cos_theta = OldMomentum*theRecoveredNormal ; 

    G4double energy = thePhotonMomentum ; 
    G4double wavelength = twopi*hbarc/energy ;
    G4double energy_eV = energy/eV ;
    G4double wavelength_nm = wavelength/nm ; 

    const G4VTouchable* touch = aTrack.GetTouchable();    
    int replicaNumber = U4Touchable::ReplicaNumber(touch);  // aka pmtid
    int pmtcat = J::DEFAULT_CAT ;    // TODO: add J API to access pmtcat+qe from replicaNumber
    double _qe = 0.0 ; 


    StackSpec<double,4> spec ; 
    spec.ls[0].d = 0. ; 
    spec.ls[1].d = parameter_accessor->get_thickness_nm( pmtcat, J::L1 );  
    spec.ls[2].d = parameter_accessor->get_thickness_nm( pmtcat, J::L2 );  
    spec.ls[3].d = 0. ; 

    spec.ls[0].nr = parameter_accessor->get_rindex( pmtcat, J::L0, J::RINDEX, energy_eV );  
    spec.ls[0].ni = parameter_accessor->get_rindex( pmtcat, J::L0, J::KINDEX, energy_eV );

    spec.ls[1].nr = parameter_accessor->get_rindex( pmtcat, J::L1, J::RINDEX, energy_eV );
    spec.ls[1].ni = parameter_accessor->get_rindex( pmtcat, J::L1, J::KINDEX, energy_eV );

    spec.ls[2].nr = parameter_accessor->get_rindex( pmtcat, J::L2, J::RINDEX, energy_eV );  
    spec.ls[2].ni = parameter_accessor->get_rindex( pmtcat, J::L2, J::KINDEX, energy_eV );  

    spec.ls[3].nr = parameter_accessor->get_rindex( pmtcat, J::L3, J::RINDEX, energy_eV );  
    spec.ls[3].ni = parameter_accessor->get_rindex( pmtcat, J::L3, J::KINDEX, energy_eV );


    Stack<double,4> stack(      wavelength_nm, minus_cos_theta, spec );  

    // NB stack is flipped for minus_cos_theta > 0. so:
    //
    //    stack.ll[0] always incident side
    //    stack.ll[3] always transmission side 

    const double _si = stack.ll[0].st.real() ; 
    const double _si2 = sqrtf( 1. - minus_cos_theta*minus_cos_theta ); 

    double E_s2 = _si > 0. ? (OldPolarization*OldMomentum.cross(theRecoveredNormal))/_si : 0. ; 
    E_s2 *= E_s2;      

    double one = 1.0 ; 
    double S = E_s2 ; 
    double P = one - S ; 

    // E_s2 : S-vs-P power fraction : signs make no difference as squared
    // E_s2 matches E1_perp*E1_perp see sysrap/tests/stmm_vs_sboundary_test.cc 

    LOG(LEVEL)
        << " count " << count 
        << " replicaNumber " << replicaNumber
        << " _si " << std::fixed << std::setw(10) << std::setprecision(5) << _si 
        << " _si2 " << std::fixed << std::setw(10) << std::setprecision(5) << _si2 
        << " theRecoveredNormal " << theRecoveredNormal 
        << " OldPolarization*OldMomentum.cross(theRecoveredNormal) " << OldPolarization*OldMomentum.cross(theRecoveredNormal) 
        << " E_s2 " << std::fixed << std::setw(10) << std::setprecision(5) << E_s2 
        ;    

    double T = S*stack.art.T_s + P*stack.art.T_p ;  // matched with TransCoeff see sysrap/tests/stmm_vs_sboundary_test.cc
    double R = S*stack.art.R_s + P*stack.art.R_p ;
    double A = one - (T+R);

    theTransmittance = T ; 
    theReflectivity  = R ; 

    LOG(LEVEL)
        << " count " << count 
        << " S " << std::fixed << std::setw(10) << std::setprecision(5) << S 
        << " P " << std::fixed << std::setw(10) << std::setprecision(5) << P
        << " T " << std::fixed << std::setw(10) << std::setprecision(5) << T 
        << " R " << std::fixed << std::setw(10) << std::setprecision(5) << R
        << " A " << std::fixed << std::setw(10) << std::setprecision(5) << A
        ;    

    // stackNormal is not flipped (as minus_cos_theta is fixed at -1.) presumably this is due to _qe definition
    Stack<double,4> stackNormal(wavelength_nm, -1. , spec ); 

    double An = one - (stackNormal.art.T + stackNormal.art.R) ; 
    double D = _qe/An;   // LOOKS STRANGE TO DIVIDE BY An  : HMM MAYBE NEED TO DIVIDE BY A TOO ? 
 
    theEfficiency = D ; 

    LOG_IF(error, D > 1.)
        << " ERR: D > 1. : " << D 
        << " _qe " << _qe
        << " An " << An
        ;

    return 'Y' ; 
}



