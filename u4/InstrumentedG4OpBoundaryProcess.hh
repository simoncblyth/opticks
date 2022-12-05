//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
//
// $Id: InstrumentedG4OpBoundaryProcess.hh 104458 2017-05-31 15:54:11Z gcosmo $
//
// 
////////////////////////////////////////////////////////////////////////
// Optical Photon Boundary Process Class Definition
////////////////////////////////////////////////////////////////////////
//
// File:        InstrumentedG4OpBoundaryProcess.hh
// Description: Discrete Process -- reflection/refraction at
//                                  optical interfaces
// Version:     1.1
// Created:     1997-06-18
// Modified:    2005-07-28 add G4ProcessType to constructor
//              1999-10-29 add method and class descriptors
//              1999-10-10 - Fill NewMomentum/NewPolarization in 
//                           DoAbsorption. These members need to be
//                           filled since DoIt calls 
//                           aParticleChange.SetMomentumChange etc.
//                           upon return (thanks to: Clark McGrew)
//              2006-11-04 - add capability of calculating the reflectivity
//                           off a metal surface by way of a complex index
//                           of refraction - Thanks to Sehwook Lee and John
//                           Hauptman (Dept. of Physics - Iowa State Univ.)
//              2009-11-10 - add capability of simulating surface reflections
//                           with Look-Up-Tables (LUT) containing measured
//                           optical reflectance for a variety of surface
//                           treatments - Thanks to Martin Janecek and
//                           William Moses (Lawrence Berkeley National Lab.)
//              2013-06-01 - add the capability of simulating the transmission
//                           of a dichronic filter
//              2017-02-24 - add capability of simulating surface reflections
//                           with Look-Up-Tables (LUT) developed in DAVIS
//
// Author:      Peter Gumplinger
//              adopted from work by Werner Keil - April 2/96
// mail:        gum@triumf.ca
//
////////////////////////////////////////////////////////////////////////

#ifndef InstrumentedG4OpBoundaryProcess_h
#define InstrumentedG4OpBoundaryProcess_h 1

/////////////
// Includes
/////////////

#include "globals.hh"
#include "templates.hh"
#include "geomdefs.hh"
#include "Randomize.hh"

#include "G4RandomTools.hh"
#include "G4RandomDirection.hh"


#include "U4RandomTools.hh"
#include "U4RandomDirection.hh"
#include "plog/Severity.h"

#ifdef WITH_PMTFASTSIM
#include "SOpBoundaryProcess.hh"
struct JPMT ; 
struct CustomBoundary ; 
#endif

#include "G4Step.hh"
#include "G4VDiscreteProcess.hh"
#include "G4DynamicParticle.hh"
#include "G4Material.hh"
#include "G4LogicalBorderSurface.hh"
#include "G4LogicalSkinSurface.hh"
#include "G4OpticalSurface.hh"
#include "G4OpticalPhoton.hh"
#include "G4TransportationManager.hh"

// Class Description:
// Discrete Process -- reflection/refraction at optical interfaces.
// Class inherits publicly from G4VDiscreteProcess.
// Class Description - End:

/////////////////////
// Class Definition
/////////////////////

enum G4OpBoundaryProcessStatus {  Undefined,
                                  Transmission, FresnelRefraction,
                                  FresnelReflection, TotalInternalReflection,
                                  LambertianReflection, LobeReflection,
                                  SpikeReflection, BackScattering,
                                  Absorption, Detection, NotAtBoundary,
                                  SameMaterial, StepTooSmall, NoRINDEX,
                                  PolishedLumirrorAirReflection,
                                  PolishedLumirrorGlueReflection,
                                  PolishedAirReflection,
                                  PolishedTeflonAirReflection,
                                  PolishedTiOAirReflection,
                                  PolishedTyvekAirReflection,
                                  PolishedVM2000AirReflection,
                                  PolishedVM2000GlueReflection,
                                  EtchedLumirrorAirReflection,
                                  EtchedLumirrorGlueReflection,
                                  EtchedAirReflection,
                                  EtchedTeflonAirReflection,
                                  EtchedTiOAirReflection,
                                  EtchedTyvekAirReflection,
                                  EtchedVM2000AirReflection,
                                  EtchedVM2000GlueReflection,
                                  GroundLumirrorAirReflection,
                                  GroundLumirrorGlueReflection,
                                  GroundAirReflection,
                                  GroundTeflonAirReflection,
                                  GroundTiOAirReflection,
                                  GroundTyvekAirReflection,
                                  GroundVM2000AirReflection,
                                  GroundVM2000GlueReflection,
                                  Dichroic };



#ifdef DEBUG_TAG
#include "U4Stack.h"
#include "SEvt.hh"
#endif



#include "U4_API_EXPORT.hh"
class U4_API InstrumentedG4OpBoundaryProcess 
    : 
    public G4VDiscreteProcess
#ifdef WITH_PMTFASTSIM
    ,public SOpBoundaryProcess
#endif
{
        int PostStepDoIt_count ; 

#ifdef DEBUG_PIDX
        static const int  PIDX ;
        int pidx ; 
        bool pidx_dump ; 
#endif

public:
        static const plog::Severity LEVEL ; 

#ifdef WITH_PMTFASTSIM
        CustomBoundary* m_custom ; 

        double getU0() const ; 
        int getU0_idx() const ; 
#endif

#ifdef DEBUG_TAG
        static const bool FLOAT ;  
        void ResetNumberOfInteractionLengthLeft();
#endif

        ////////////////////////////////
        // Constructors and Destructor
        ////////////////////////////////

        InstrumentedG4OpBoundaryProcess(const G4String& processName = "OpBoundary",
                                     G4ProcessType type = fOptical);
        ~InstrumentedG4OpBoundaryProcess();

private:

        InstrumentedG4OpBoundaryProcess(const InstrumentedG4OpBoundaryProcess &right);

        //////////////
        // Operators
        //////////////

        InstrumentedG4OpBoundaryProcess& operator=(const InstrumentedG4OpBoundaryProcess &right);

public:

	////////////
        // Methods
        ////////////

        G4bool IsApplicable(const G4ParticleDefinition& aParticleType);
        // Returns true -> 'is applicable' only for an optical photon.

        G4double GetMeanFreePath(const G4Track& ,
                                 G4double ,
                                 G4ForceCondition* condition);
        // Returns infinity; i. e. the process does not limit the step,
        // but sets the 'Forced' condition for the DoIt to be invoked at
        // every step. However, only at a boundary will any action be
        // taken.

        G4VParticleChange* PostStepDoIt(const G4Track& aTrack,
                                        const G4Step&  aStep);

        G4VParticleChange* PostStepDoIt_(const G4Track& aTrack,
                                         const G4Step&  aStep);

        // This is the method implementing boundary processes.

        G4OpBoundaryProcessStatus GetStatus() const;
        // Returns the current status.

        void SetInvokeSD(G4bool );
        // Set flag for call to InvokeSD method.

private:

        G4bool G4BooleanRand(const G4double prob) const;
        G4bool G4BooleanRand_TransCoeff(const G4double prob) const;
        G4bool G4BooleanRand_theEfficiency(const G4double prob) const;
        G4bool G4BooleanRand_theReflectivity(const G4double prob) const;

        G4ThreeVector GetFacetNormal(const G4ThreeVector& Momentum,
                                     const G4ThreeVector&  Normal) const;

        void DielectricMetal();
        void DielectricDielectric();

        void DielectricLUT();
        void DielectricLUTDAVIS();

        void DielectricDichroic();

        void ChooseReflection();
        void DoAbsorption();
        void DoReflection();

        G4double GetIncidentAngle();
        // Returns the incident angle of optical photon

        G4double GetReflectivity(G4double E1_perp,
                                 G4double E1_parl,
                                 G4double incidentangle,
                                 G4double RealRindex,
                                 G4double ImaginaryRindex);
        // Returns the Reflectivity on a metalic surface

        void CalculateReflectivity(void);

        void BoundaryProcessVerbose(void) const;

        // Invoke SD for post step point if the photon is 'detected'
        G4bool InvokeSD(const G4Step* step);

#ifdef WITH_PMTFASTSIM
        char CustomART(const G4Track& aTrack, const G4Step& aStep ); 
#endif
public:  
        static void Save(const char* dir); 

#ifdef WITH_PMTFASTSIM
        JPMT*  m_jpmt ; 
        int    m_CustomART_count ; 
        double m_u0 ; 
        int    m_u0_idx ; 
        int    m_u0_idx_delta ; 
#endif

private:
        G4double thePhotonMomentum;

        G4ThreeVector OldMomentum;
        G4ThreeVector OldPolarization;

        G4ThreeVector NewMomentum;
        G4ThreeVector NewPolarization;

        G4ThreeVector theGlobalPoint ;
        G4ThreeVector theGlobalExitNormal;   // SCB: as provided by G4Navigator 
        G4ThreeVector theGlobalNormal;
        G4ThreeVector theRecoveredNormal ;   // SCB: try to use haveEnteredDaughter status to unflip back to G4Track independent normal 

        G4ThreeVector theFacetNormal;

        G4Material* Material1;
        G4Material* Material2;

        G4OpticalSurface* OpticalSurface;

        G4MaterialPropertyVector* PropertyPointer;
        G4MaterialPropertyVector* PropertyPointer1;
        G4MaterialPropertyVector* PropertyPointer2;

        G4double Rindex1;
        G4double Rindex2;

        G4double cost1, cost2, sint1, sint2;

        G4OpBoundaryProcessStatus theStatus;

        G4OpticalSurfaceModel theModel;

        G4OpticalSurfaceFinish theFinish;

        G4double theReflectivity;
        G4double theEfficiency;
        G4double theTransmittance;

        G4double theSurfaceRoughness;

        G4double prob_sl, prob_ss, prob_bs;

        G4int iTE, iTM;

        G4double kCarTolerance;

        size_t idx, idy;
        G4Physics2DVector* DichroicVector;

        G4bool fInvokeSD;
};



#endif /* InstrumentedG4OpBoundaryProcess_h */
