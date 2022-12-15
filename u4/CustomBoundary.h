#pragma once
/**
CustomBoundary.h : Adding Multi-Layer TMM ARTD to G4OpBoundaryProcess
=======================================================================

Alternative names::

    CustomSurface.h
    TMMSurface.h
    MultiLayerSurface.h 
    MultiFilmSurface.h 

Incorporating "Surface" in the name seems more appropriate than Boundary
as its acting as a "smart" surface from Geant4 point of view.


This extends G4OpBoundaryProcess providing custom calculation of ARTD (Absorb,
Reflect, Transmit, Detect) probabilities.  This can be used for example to add
multi-layer TMM calculation of ARTD based on layer refractive indices and
thicknesses. 

For example this could implement coherent interference effects for a PMT
on the boundary between Pyrex and Vacuum where thin layers of ARC 
(anti-reflection coating) and PHC (photocathode) are located::

         +---------------pmt-Pyrex----------------+
         |                                        |    
         |                                        |    
         |                                        |    
         |       +~inner~Vacuum~~~~~~~~~~~+       |    
         |       !                        !       |    
         |       !                        !       |    
         |       !                        !       |    
         |       !                        !       |    
         |       !                        !       |    
         |       +                        +       |    
         |       |                        |       |    
         |       |                        |       |    
         |       |                        |       |    
         |       |                        |       |    
         |       |                        |       |    
         |       +------------------------+       |    
         |                                        |    
         |                                        |    
         |                                        |    
         +----------------------------------------+

The parameter names follow those of G4OpBoundaryProcess apart from
*theRecoveredNormal*.  *theRecoveredNormal* is required to be the outwards
normal for the inner volume in global frame. Absolutely no G4Track dependent
flips must be applied to this normal, it must be purely an outwards geometry
normal. NB this means that theGlobalExitNormal must be corrected, removing a
flip done by G4Navigator based on "enteredDaughter".  

For example, from a modified G4OpBoundaryProcess::

     418     G4bool haveEnteredDaughter= (thePostPV->GetMotherLogical() == thePrePV ->GetLogicalVolume()); // SCB
     419 
     ...
     449     G4int hNavId = G4ParallelWorldProcess::GetHypNavigatorID();
     450     std::vector<G4Navigator*>::iterator iNav =
     451                 G4TransportationManager::GetTransportationManager()->
     452                                          GetActiveNavigatorsIterator();
     453     theGlobalExitNormal =
     454                    (iNav[hNavId])->GetGlobalExitNormal(theGlobalPoint,&valid);
     455 
     456     // theGlobalExitNormal is already oriented by G4Navigator to point from vol1 -> vol2 
     457     // so undo that flip by G4Navigator in order to recover the original geometry 
     458     // normal that is independent of G4Track direction
     459 
     460     theRecoveredNormal = ( haveEnteredDaughter ? -1. : 1. )* theGlobalExitNormal  ;
     464     theGlobalNormal = theGlobalExitNormal ;

CAUTION: theGlobalNormal is compromised (by being flipped) multiple times 
within G4OpBoundaryProcess so it is vital that a separate *theGlobalExitNormal* 
is used which is never compromised. 

Requiring *theRecoveredNormal* allows the *doIt* to be implemented without
having to transform into local frame, because it provides an absolute
orientation. If this approach turns out to be inconvenient another way of
establishing an absolute orientation is to take a compromised normal (that has
been flipped depending on G4Track an unknown number of times) and transform
that into the local frame and then fix the orientation based on the sign of the
z-component of the local normal::

    G4ThreeVector localNormal = transform.TransformAxis(theGlobalNormal);
    bool outwards = localNormal.z() > 0. ; // as always upper hemi of PMT in local frame  
    G4ThreeVector surface_normal = (outwards ? 1. : -1.)*localNormal ; 

The disadvantage of using a local normal is that it then makes it necessary to
transform momentum and polarization into local frame and transform back to
global frame.  Conversely using *theRecoveredNormal* which is an absolute
global normal allows the whole calculation to be done in global frame, avoiding
all that transforming. 


CustomBoundary::maybe_doIt
    for OpticalSurfaceName starting with '@' local_z is used to 
    check for upper half (eg upper hemisphere of ellipsoidal PMT) in which 
    case the doIt is run. The local_z is obtained from theGlobalPoint transformed 
    into the frame of the G4Track. 
     
CustomBoundary::doIt 
    performs its calculations purely in global frame, 
    using the dot product between *theRecoveredNormal* and *OldMomentum* to provide the
    orientation of the photon with respect to the boundary and the *minus_cos_theta* 
    where theta is the angle of incidence. 

template type J
    required to have enumerations : RINDEX, KINDEX, L0, L1, L2, L3, DEFAULT_CAT
    and methods to access layer refractive indices and thicknesses for various 
    categories of sensors (eg PMTs) and layers with arguments including energy_eV  
  
    * currently required to have argumentless ctor 


WIP : polarization comparison sysrap/tests/stmm_vs_sboundary_test.sh 
----------------------------------------------------------------------

The polarization on reflection copied from junoPMTOpticalModel is for TIR 
(not ordinary Fresnel Reflection)

WIP : Can standard DielectricDielectric be used with TMM calc ?
--------------------------------------------------------------------------------

Trying to reuse the standard mom/pol boundary impl together with 
the overidden calculation of the below, done in CustomART::

   theTransmittance
   theReflectivity 
   theEfficiency

* will require restriction to specific theModel and theFinish 
  values to avoid the code wandering off the needed path

WIP : Detector Specific property rindex + thickness + qe access
------------------------------------------------------------------

* maybe make template type J accept an MPT MaterialPropertiesTable in its ctor, enabling 
  it to be implemented purely from a bunch of custom properties loading into the 
  OpticalSurface : then it would be more convenient for the instance of J to 
  be passed in as a parameter to CustomBoundary. 

**/

#include <cstring>

struct CustomStatus
{
   static constexpr const char* ACTIVE = "ARTD" ; 
   static constexpr const char* A_ = "Absorb" ; 
   static constexpr const char* R_ = "Reflect" ; 
   static constexpr const char* T_ = "Transmit" ; 
   static constexpr const char* D_ = "Detect" ; 
   static const char* Name(char status); 
}; 

inline const char* CustomStatus::Name(char status)
{
   const char* s = nullptr ; 
   switch(status)
   {
       case 'A': s = A_ ; break ; 
       case 'R': s = R_ ; break ; 
       case 'T': s = T_ ; break ; 
       case 'D': s = D_ ; break ; 
   }
   return s ; 
}


#include "G4ThreeVector.hh"
#include "G4TwoVector.hh"
#include "Randomize.hh"

#include "SLOG.hh"
#include "JPMT.h"
#include "Layr.h"
#include "U4Touchable.h"

#ifdef DEBUG_TAG
#include "U4UniformRand.h"
#include "SPhoton_Debug.h"
template<> std::vector<SPhoton_Debug<'B'>> SPhoton_Debug<'B'>::record = {} ;
#endif

/**
CustomART
-----------

Trying to do less than CustomBoundary, instead just calculates::

    theTransmittance
    theReflectivity
    theEfficiency 

with the mechanics of deciding on R/T/A/D and 
changing photon momentum, polarization, theStatus 
done by the nearly "standard" G4OpBoundaryProcess.   

**/
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

    const G4ThreeVector& OldMomentum ; 
    const G4ThreeVector& OldPolarization ; 
    const G4ThreeVector& theGlobalPoint ; 
    const G4ThreeVector& theRecoveredNormal ; 
    const G4double& thePhotonMomentum ; 

    CustomART(
        G4double& theTransmittance,
        G4double& theReflectivity,
        G4double& theEfficiency,
        const G4ThreeVector& OldMomentum,  
        const G4ThreeVector& OldPolarization,
        const G4ThreeVector& theGlobalPoint,
        const G4ThreeVector& theRecoveredNormal,
        const G4double& thePhotonMomentum
    );  

    G4bool maybe_doIt(const char* OpticalSurfaceName, const G4Track& aTrack, const G4Step& aStep) ;  
    void   doIt(const G4Track& aTrack, const G4Step& aStep ); 
}; 

template<typename J>
inline CustomART<J>::CustomART(
          G4double& theTransmittance_,
          G4double& theReflectivity_,
          G4double& theEfficiency_,
    const G4ThreeVector& OldMomentum_,
    const G4ThreeVector& OldPolarization_,
    const G4ThreeVector& theGlobalPoint_, 
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
    OldMomentum(OldMomentum_),
    OldPolarization(OldPolarization_),
    theGlobalPoint(theGlobalPoint_),
    theRecoveredNormal(theRecoveredNormal_),
    thePhotonMomentum(thePhotonMomentum_) 
{
}

template<typename J>
inline G4bool CustomART<J>::maybe_doIt(const char* OpticalSurfaceName, const G4Track& aTrack, const G4Step& aStep )
{
    if( OpticalSurfaceName == nullptr || OpticalSurfaceName[0] != '@') return false ; 

    const G4AffineTransform& transform = aTrack.GetTouchable()->GetHistory()->GetTopTransform();
    G4ThreeVector localPoint = transform.TransformPoint(theGlobalPoint);
    zlocal = localPoint.z() ; 
    lposcost = localPoint.cosTheta() ;  
    // Q:What is lposcost for ?  
    // A:Preparing for doing this on GPU, as lposcost is available there already but not zlocal 

    bool zlocal_positive = zlocal > 0. ;   
    if(zlocal_positive)  doIt(aTrack, aStep) ;
    return zlocal_positive ; 
}

template<typename J>
inline void CustomART<J>::doIt(const G4Track& aTrack, const G4Step& aStep )
{
    G4double minus_cos_theta = OldMomentum*theRecoveredNormal ; 
    G4double orientation = minus_cos_theta < 0. ? 1. : -1.  ; 
    G4ThreeVector oriented_normal = orientation*theRecoveredNormal ;

    G4double energy = thePhotonMomentum ; 
    G4double wavelength = twopi*hbarc/energy ;

    G4double energy_eV = energy/eV ;
    G4double wavelength_nm = wavelength/nm ; 

    const G4VTouchable* touch = aTrack.GetTouchable();    
    int replicaNumber = U4Touchable::ReplicaNumber(touch);  // aka pmtid
    // TODO: add J API to access pmtcat+qe from replicaNumber
    int pmtcat = J::DEFAULT_CAT ; 
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

    double E_s2 = _si > 0. ? (OldPolarization*OldMomentum.cross(theRecoveredNormal))/_si : 0. ; 
    E_s2 *= E_s2;      

    // E_s2 : S-vs-P power fraction : signs make no difference as squared
    // E_s2 matches E1_perp*E1_perp see sysrap/tests/stmm_vs_sboundary_test.cc 

    LOG(LEVEL)
        << " count " << count 
        << " replicaNumber " << replicaNumber
        << " _si " << std::fixed << std::setw(10) << std::setprecision(5) << _si 
        << " theRecoveredNormal " << theRecoveredNormal 
        << " OldPolarization*OldMomentum.cross(theRecoveredNormal) " << OldPolarization*OldMomentum.cross(theRecoveredNormal) 
        << " E_s2 " << std::fixed << std::setw(10) << std::setprecision(5) << E_s2 
        ;    

    double fT_s = stack.art.T_s ; 
    double fT_p = stack.art.T_p ; 
    double fR_s = stack.art.R_s ; 
    double fR_p = stack.art.R_p ; 

    double one = 1.0 ; 
    double T = fT_s*E_s2 + fT_p*(one-E_s2);  // matched with TransCoeff see sysrap/tests/stmm_vs_sboundary_test.cc
    double R = fR_s*E_s2 + fR_p*(one-E_s2);
    double A = one - (T+R);

    theTransmittance = T ; 
    theReflectivity = R ; 

    LOG(LEVEL)
        << " count " << count 
        << " E_s2 " << std::fixed << std::setw(10) << std::setprecision(5) << E_s2 
        << " fT_s " << std::fixed << std::setw(10) << std::setprecision(5) << fT_s 
        << " 1-E_s2 " << std::fixed << std::setw(10) << std::setprecision(5) << (1.-E_s2)
        << " fT_p " << std::fixed << std::setw(10) << std::setprecision(5) << fT_p 
        << " T " << std::fixed << std::setw(10) << std::setprecision(5) << T 
        ;    

    LOG(LEVEL)
        << " count " << count 
        << " E_s2 " << std::fixed << std::setw(10) << std::setprecision(5) << E_s2 
        << " fR_s " << std::fixed << std::setw(10) << std::setprecision(5) << fR_s 
        << " 1-E_s2 " << std::fixed << std::setw(10) << std::setprecision(5) << (1.-E_s2)
        << " fR_p " << std::fixed << std::setw(10) << std::setprecision(5) << fR_p 
        << " R " << std::fixed << std::setw(10) << std::setprecision(5) << R 
        << " A " << std::fixed << std::setw(10) << std::setprecision(5) << A 
        ;  

    // stackNormal is not flipped (as minus_cos_theta is fixed at -1.) presumably this is due to _qe definition

    Stack<double,4> stackNormal(wavelength_nm, -1. , spec ); 

    double An = one - (stackNormal.art.T + stackNormal.art.R) ; 
    double D = _qe/An;   // LOOKS STRANGE TO DIVIDE BY An 
 
    theEfficiency = D ; 

    LOG_IF(error, D > 1.)
        << " ERR: D > 1. : " << D 
        << " _qe " << _qe
        << " An " << An
        ;

}







template<typename J>
struct CustomBoundary
{
    static const constexpr plog::Severity LEVEL = debug ;  
    static void Save(const char* fold); 

    J*     parameter_accessor ; 
    int    count ; 
    double zlocal ; 
    double lposcost ; 
    char   customStatus ; 

    G4ThreeVector& NewMomentum ; 
    G4ThreeVector& NewPolarization ; 
    G4ParticleChange& aParticleChange ; 
    G4OpBoundaryProcessStatus& theStatus ;

    const G4ThreeVector& OldMomentum ; 
    const G4ThreeVector& OldPolarization ; 
    const G4ThreeVector& theGlobalPoint ; 
    const G4ThreeVector& theRecoveredNormal ; 
    const G4double& thePhotonMomentum ; 

    CustomBoundary( 
        G4ThreeVector& NewMomentum,
        G4ThreeVector& NewPolarization,
        G4ParticleChange& aParticleChange,
        G4OpBoundaryProcessStatus& theStatus,
        const G4ThreeVector& OldMomentum,  
        const G4ThreeVector& OldPolarization,
        const G4ThreeVector& theGlobalPoint,
        const G4ThreeVector& theRecoveredNormal,
        const G4double& thePhotonMomentum
      ); 

    G4bool maybe_doIt(const char* OpticalSurfaceName, const G4Track& aTrack, const G4Step& aStep) ;  
    char doIt(const G4Track& aTrack, const G4Step& aStep ); 
};


template<typename J>
inline void CustomBoundary<J>::Save(const char* fold) // static
{
    SPhoton_Debug<'B'>::Save(fold);   
}

template<typename J>
inline CustomBoundary<J>::CustomBoundary(
          G4ThreeVector& NewMomentum_,
          G4ThreeVector& NewPolarization_,
          G4ParticleChange& aParticleChange_,
          G4OpBoundaryProcessStatus& theStatus_,
    const G4ThreeVector& OldMomentum_,
    const G4ThreeVector& OldPolarization_,
    const G4ThreeVector& theGlobalPoint_, 
    const G4ThreeVector& theRecoveredNormal_,
    const G4double&      thePhotonMomentum_
    )
    :
    parameter_accessor(new J),
    count(0),
    zlocal(-1.),
    lposcost(-2.),
    customStatus('?'),
    NewMomentum(NewMomentum_),
    NewPolarization(NewPolarization_),
    aParticleChange(aParticleChange_),
    theStatus(theStatus_),
    OldMomentum(OldMomentum_),
    OldPolarization(OldPolarization_),
    theGlobalPoint(theGlobalPoint_),
    theRecoveredNormal(theRecoveredNormal_),
    thePhotonMomentum(thePhotonMomentum_) 
{
}

template<typename J>
inline G4bool CustomBoundary<J>::maybe_doIt(const char* OpticalSurfaceName, const G4Track& aTrack, const G4Step& aStep )
{
    customStatus = 'N' ; // N:OpticalSurfaceName does not start with '@'
    if( OpticalSurfaceName == nullptr || OpticalSurfaceName[0] != '@') return false ; 

    const G4AffineTransform& transform = aTrack.GetTouchable()->GetHistory()->GetTopTransform();
    G4ThreeVector localPoint = transform.TransformPoint(theGlobalPoint);
    zlocal = localPoint.z() ; 
    lposcost = localPoint.cosTheta() ; 

    customStatus = zlocal > 0. ? doIt(aTrack, aStep) : 'Z' ;  // Z:-ve Z not triggered
    bool doneIt = strchr("ARTD", customStatus) != nullptr ; 
    return doneIt ; 
}


template<typename J>
inline char CustomBoundary<J>::doIt(const G4Track& aTrack, const G4Step& aStep )
{
    const G4ThreeVector& surface_normal = theRecoveredNormal ; 
    const G4ThreeVector& direction      = OldMomentum ; 
    const G4ThreeVector& polarization   = OldPolarization ; 
  
    G4double minus_cos_theta = direction*surface_normal ; 
    G4double orientation = minus_cos_theta < 0. ? 1. : -1.  ; 
    G4ThreeVector oriented_normal = orientation*surface_normal ;

    G4double energy = thePhotonMomentum ; 
    G4double wavelength = twopi*hbarc/energy ;

    G4double energy_eV = energy/eV ;
    G4double wavelength_nm = wavelength/nm ; 

    const G4VTouchable* touch = aTrack.GetTouchable();    
    int replicaNumber = U4Touchable::ReplicaNumber(touch);  // aka pmtid
    // TODO: add J API to access pmtcat+qe from replicaNumber
    int pmtcat = J::DEFAULT_CAT ; 
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

    const double _ni = stack.ll[0].n.real() ; 
    const double _si = stack.ll[0].st.real() ; 
    const double _ci = stack.ll[0].ct.real() ;

    const double _nt = stack.ll[3].n.real() ; 
    const double _ct = stack.ll[3].ct.real() ;
    const double eta = _ni/_nt ;  

    Stack<double,4> stackNormal(wavelength_nm, -1.            , spec ); 
    // stackNormal is not flipped (as minus_cos_theta is fixed at -1.) 
    //      presumably this is due to _qe definition


    // E_s2 : S-vs-P power fraction : signs make no difference as squared
    double E_s2 = _si > 0. ? (polarization*direction.cross(oriented_normal))/_si : 0. ; 
    E_s2 *= E_s2;      // E_s2 matches E1_perp*E1_perp see sysrap/tests/stmm_vs_sboundary_test.cc 

    LOG(LEVEL)
        << " count " << count 
        << " replicaNumber " << replicaNumber
        << " _si " << std::fixed << std::setw(10) << std::setprecision(5) << _si 
        << " oriented_normal " << oriented_normal 
        << " polarization*direction.cross(oriented_normal) " << polarization*direction.cross(oriented_normal) 
        << " E_s2 " << std::fixed << std::setw(10) << std::setprecision(5) << E_s2 
        ;    

    double fT_s = stack.art.T_s ; 
    double fT_p = stack.art.T_p ; 
    double fR_s = stack.art.R_s ; 
    double fR_p = stack.art.R_p ; 
    double one = 1.0 ; 
    double T = fT_s*E_s2 + fT_p*(one-E_s2);  // matched with TransCoeff see sysrap/tests/stmm_vs_sboundary_test.cc
    double R = fR_s*E_s2 + fR_p*(one-E_s2);
    double A = one - (T+R);


   LOG(LEVEL)
        << " count " << count 
        << " E_s2 " << std::fixed << std::setw(10) << std::setprecision(5) << E_s2 
        << " fT_s " << std::fixed << std::setw(10) << std::setprecision(5) << fT_s 
        << " 1-E_s2 " << std::fixed << std::setw(10) << std::setprecision(5) << (1.-E_s2)
        << " fT_p " << std::fixed << std::setw(10) << std::setprecision(5) << fT_p 
        << " T " << std::fixed << std::setw(10) << std::setprecision(5) << T 
        ;    

    LOG(LEVEL)
        << " count " << count 
        << " E_s2 " << std::fixed << std::setw(10) << std::setprecision(5) << E_s2 
        << " fR_s " << std::fixed << std::setw(10) << std::setprecision(5) << fR_s 
        << " 1-E_s2 " << std::fixed << std::setw(10) << std::setprecision(5) << (1.-E_s2)
        << " fR_p " << std::fixed << std::setw(10) << std::setprecision(5) << fR_p 
        << " R " << std::fixed << std::setw(10) << std::setprecision(5) << R 
        << " A " << std::fixed << std::setw(10) << std::setprecision(5) << A 
        ;    


    double fT_n = stackNormal.art.T ; 
    double fR_n = stackNormal.art.R ; 
    double An = one - (fT_n+fR_n);
    double D = _qe/An;

    LOG_IF(error, D > 1.)
         << " ERR: D > 1. : " << D 
         << " _qe " << _qe
         << " An " << An
         ;

    double u0 = G4UniformRand();
    double u1 = G4UniformRand();

    /**
       0         A         A+R         1
       |---------+----------+----------|  u0
          D/A         R          T
          u1 
    **/

    char status =  u0 < A ? ( u1 < D ? 'D' : 'A' ) : ( u0 < A + R ? 'R' : 'T' ) ; 



#ifdef DEBUG_TAG
    int u0_idx = U4UniformRand::Find(u0, SEvt::UU);     
    int u1_idx = U4UniformRand::Find(u1, SEvt::UU);     

    LOG(LEVEL) 
         << " u0 " << U4UniformRand::Desc(u0) 
         << " u0_idx " << u0_idx 
         << " A "   << std::setw(10) << std::fixed << std::setprecision(4) << A
         << " A+R " << std::setw(10) << std::fixed << std::setprecision(4) << (A+R) 
         << " T "   << std::setw(10) << std::fixed << std::setprecision(4) << T
         << " status " 
         << status 
         << " DECISION "
         ; 
    LOG(LEVEL) 
         << " u1 " << U4UniformRand::Desc(u1) 
         << " u1_idx " << u1_idx 
         << " D " << std::setw(10) << std::fixed << std::setprecision(4) << D
         ; 
#endif


    NewMomentum = OldMomentum ; 
    NewPolarization = OldPolarization ; 

    // The below reflect/refract is a copy of junoPMTOpticalModel  
    //
    //    NewMomentum 
    //        looks OK
    //
    //    NewPolarization 
    //        looks to be guess(aka wrong) 
    //        which does not follow along from the above S vs P fractions 
    //
    //        polarization should depend on S and P fractions (think Brewsters angle) 
    //        polarization vector does not reflect like the momentum vector does 
    // 
    // comparing with G4OpBoundaryProcess in sysrap/tests/stmm_vs_sboundary_test.cc
    //

    if( status == 'R' )
    {
        theStatus = FresnelReflection ;
        NewMomentum   -= 2.*(OldMomentum*oriented_normal)*oriented_normal ;
        NewPolarization -= 2.*(OldPolarization*oriented_normal)*oriented_normal ;

       // THIS IS NOW G4OpBoundaryProcess CHANGES POLARIZATION FOR TOTAL INTERNAL REFLECTION, 
       // NOT FRESNEL REFLECTION

    }
    else if( status == 'T' )
    {
        theStatus = FresnelRefraction ; 
        NewMomentum = eta*OldMomentum + (eta*_ci - _ct)*oriented_normal ;  

        // NewMomentum = NewMomentum.unit() ;  
        // No need to normalize that is inherent. Derived in qsim.h:propagate_at_boundary.
        // This formula matches that used in qsim.h:propagate_at_boundary
        // The bracket is flipped compared with junoPMTOpticalModel.
        // TODO: check the oriented_normal sign convention, is it really flipped. 

        NewPolarization = (OldPolarization-(OldPolarization*OldMomentum)*OldMomentum).unit();

        // HMM: WHERE DID THIS EXPRESSION COME FROM ? 
        // Q: As light is transverse OldPolarization*OldMomentum is ZERO ? SO THE ABOVE BECOMES : NewPolarization = OldPolarization
        //    Also would expect dependency on S and P fractions ? 


    }
    else if(status == 'A' || status == 'D')
    {
        theStatus = status == 'D' ? Detection : Absorption ;

        aParticleChange.ProposeLocalEnergyDeposit(status == 'D' ? thePhotonMomentum : 0.0) ;
        aParticleChange.ProposeTrackStatus(fStopAndKill) ;
    }



#ifdef DEBUG_TAG
    SPhoton_Debug<'B'> dbg ; 

    dbg.pos = theGlobalPoint ; 
    dbg.time = aTrack.GetLocalTime();  // just for debug, the only use of aTrack 

    dbg.mom = NewMomentum ; 
    dbg.iindex = 0 ; 

    dbg.pol = NewPolarization ;  
    dbg.wavelength = wavelength_nm ; 

    dbg.nrm = theRecoveredNormal ;  
    dbg.spare = 0. ; 

    dbg.u0 = u0 ; 
    dbg.x1 = 0. ; 
    dbg.x2 = 0. ; 
    dbg.u0_idx = 0 ; 

    LOG(LEVEL)
       << " time " << dbg.time
       << " dbg.Count " << dbg.Count()
       << " dbg.Name " << dbg.Name()
       << " status " << status
       << " CustomStatus::Name(status) " << CustomStatus::Name(status)
       ;    

    dbg.add();  
#endif

    return status ; 
}


