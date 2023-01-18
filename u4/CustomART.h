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


Is 2-layer (Pyrex,Vacuum) polarization direction calc applicable to 4-layer (Pyrex,ARC,PHC,Vacuum) situation ? 
-----------------------------------------------------------------------------------------------------------------

The Geant4 calculation of polarization direction for a simple
boundary between two layers is based on continuity of E and B fields
in S and P directions at the boundary, essentially Maxwells Eqn boundary conditions.
Exactly the same thing yields Snells law::

    n1 sin t1  = n2 sin t2 

My thinking on this is that Snell's law with a bunch of layers would be::

    n1 sin t1 = n2 sin t2 =  n3 sin t3 =  n4 sin t4 

So the boundary conditions from 1->4 where n1 and n4 are real still gives::

    n1 sin t1 = n4 sin t4

Even when n2,t2,n3,t3 are complex.

So by analogy that makes me think that the 2-layer polarization calculation 
between layers 1 and 4 (as done by G4OpBoundaryProcess) 
should still be valid even when there is a stack of extra layers 
inbetween layer 1 and 4. 

Essentially the stack calculation changes A,R,T so it changes
how much things happen : but it doesnt change what happens. 
So the two-layer polarization calculation from first and last layer 
should still be valid to the situation of the stack.

Do you agree with this argument ? 

**/

#include "G4ThreeVector.hh"

#include "SLOG.hh"
#include "JPMT.h"
#include "Layr.h"       // TODO: compare with stmm.h 
#include "U4Touchable.h"

//#include "CustomStatus.h"  // simply naming status chars, non-essential 


template<typename J>
struct CustomART
{
    static const constexpr plog::Severity LEVEL = debug ;  
    J*     accessor ; 
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
    accessor(new J),
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

/**
CustomART::maybe_doIt
------------------------


Q:What is lposcost for ?  
A:Preparing for doing this on GPU, as lposcost is available there already but zlocal is not, 
  so want to check the sign of lposcost is following that of zlocal. It looks 
  like it should:: 

    157 inline double Hep3Vector::cosTheta() const {
    158   double ptot = mag();
    159   return ptot == 0.0 ? 1.0 : dz/ptot;
    160 }

**/

template<typename J>
inline char CustomART<J>::maybe_doIt(const char* OpticalSurfaceName, const G4Track& aTrack, const G4Step& aStep )
{
    if( OpticalSurfaceName == nullptr || OpticalSurfaceName[0] != '@') return 'N' ; 

    const G4AffineTransform& transform = aTrack.GetTouchable()->GetHistory()->GetTopTransform();
    G4ThreeVector localPoint = transform.TransformPoint(theGlobalPoint);
    zlocal = localPoint.z() ; 
    lposcost = localPoint.cosTheta() ;  

    if(zlocal <= 0) return 'Z' ; 

    return doIt(aTrack, aStep) ;
}

/**
CustomART<J>::doIt
--------------------

NB stack is flipped for minus_cos_theta > 0. so:

* stack.ll[0] always incident side
* stack.ll[3] always transmission side 

**/

template<typename J>
inline char CustomART<J>::doIt(const G4Track& aTrack, const G4Step& aStep )
{
    G4double minus_cos_theta = OldMomentum*theRecoveredNormal ; 

    G4double energy = thePhotonMomentum ; 
    G4double wavelength = twopi*hbarc/energy ;
    G4double energy_eV = energy/eV ;
    G4double wavelength_nm = wavelength/nm ; 

    const G4VTouchable* touch = aTrack.GetTouchable();    
    int pmtid = U4Touchable::ReplicaNumber(touch);

    int pmtcat = accessor->get_pmtcat( pmtid ) ; 
    double _qe = minus_cos_theta > 0. ? 0.0 : accessor->get_pmtid_qe( pmtid, energy ) ;  

    std::array<double,16> a_spec ; 
    accessor->get_stackspec(a_spec, pmtcat, energy_eV ); 

    StackSpec<double,4> spec ; 
    spec.import( a_spec ); 
    //LOG(info) << " spec " << std::endl << spec ; 

    Stack<double,4> stack(wavelength_nm, minus_cos_theta, spec );  

    const double _si = stack.ll[0].st.real() ; 
    const double _si2 = sqrtf( 1. - minus_cos_theta*minus_cos_theta ); 

    double E_s2 = _si > 0. ? (OldPolarization*OldMomentum.cross(theRecoveredNormal))/_si : 0. ; 
    E_s2 *= E_s2;      

    // E_s2 : S-vs-P power fraction : signs make no difference as squared
    // E_s2 matches E1_perp*E1_perp see sysrap/tests/stmm_vs_sboundary_test.cc 

    double one = 1.0 ; 
    double S = E_s2 ; 
    double P = one - S ; 

    LOG(LEVEL)
        << " count " << count 
        << " pmtid " << pmtid
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


