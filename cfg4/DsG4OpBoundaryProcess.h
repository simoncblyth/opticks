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
// $Id: DsG4OpBoundaryProcess.hh,v 1.18 2008/11/07 17:59:37 gum Exp $
// GEANT4 tag $Name: geant4-09-02 $
//
// 
////////////////////////////////////////////////////////////////////////
// Optical Photon Boundary Process Class Definition
////////////////////////////////////////////////////////////////////////
//
// File:        DsG4OpBoundaryProcess.hh
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
//
// Author:      Peter Gumplinger
//              adopted from work by Werner Keil - April 2/96
// mail:        gum@triumf.ca
//
// CVS version tag: 
////////////////////////////////////////////////////////////////////////

#ifndef DsG4OpBoundaryProcess_h
#define DsG4OpBoundaryProcess_h 1


// SCB
#define SCB_REFLECT_CHEAT 1
#define GEANT4_BT_GROUPVEL_FIX 1
//#define SCB_BND_DEBUG 1


/////////////
// Includes
/////////////

#include "globals.hh"
#include "templates.hh"
#include "geomdefs.hh"
#include "Randomize.hh"



#include "G4RandomTools.hh"
#include "G4RandomDirection.hh"

#include "G4Step.hh"
#include "G4VDiscreteProcess.hh"
#include "G4DynamicParticle.hh"
#include "G4Material.hh"
#include "G4LogicalBorderSurface.hh"
#include "G4LogicalSkinSurface.hh"
#include "G4OpticalSurface.hh"
#include "G4OpticalPhoton.hh"
#include "G4TransportationManager.hh"


#include "DsG4OpBoundaryProcessStatus.h" 

#include "CRandomEngine.hh"
#include "CG4.hh"

class CMaterialLib ; 
class Opticks ; 

// Class Description:
// Discrete Process -- reflection/refraction at optical interfaces.
// Class inherits publicly from G4VDiscreteProcess.
// Class Description - End:

/////////////////////
// Class Definition
/////////////////////
class DsG4OpBoundaryProcess : public G4VDiscreteProcess
{

private:

        //////////////
        // Operators
        //////////////

        // DsG4OpBoundaryProcess& operator=(const DsG4OpBoundaryProcess &right);

        // DsG4OpBoundaryProcess(const DsG4OpBoundaryProcess &right);

public: // Without description

        ////////////////////////////////
        // Constructors and Destructor
        ////////////////////////////////

        DsG4OpBoundaryProcess(       CG4* g4 = NULL,
                                     const G4String& processName = "OpBoundary",
                                     G4ProcessType type = fOptical);

	~DsG4OpBoundaryProcess();

	////////////
	// Methods
        ////////////

public: // With description

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
        // This is the method implementing boundary processes.

	G4OpticalSurfaceModel GetModel() const;
        // Returns the optical surface mode.

        Ds::DsG4OpBoundaryProcessStatus GetStatus() const;
        // Returns the current status.

	G4double GetIncidentAngle();
        // Returns the incident angle of optical photon

	G4double GetReflectivity(G4double E1_perp,
                                 G4double E1_parl,
                                 G4double incidentangle,
	                         G4double RealRindex,
                                 G4double ImaginaryRindex);
        // Returns the Reflectivity on a metalic surface

	void           SetModel(G4OpticalSurfaceModel model);
	// Set the optical surface model to be followed
        // (glisur || unified).

private:

	G4bool G4BooleanRand(const G4double prob) ;

	G4ThreeVector GetFacetNormal(const G4ThreeVector& Momentum,
				     const G4ThreeVector&  Normal) const;

	void DielectricMetal();
	void DielectricDielectric();

	void ChooseReflection(); 
      //                                         u -> theStatus 
      //
      //                                 0:prob_ss -> SpikeReflection
      //                   prob_ss:prop_ss+prob_sl -> LobeReflection
      //   prob_ss+prob_sl:prob_ss+prob_sl+prob_bs -> BackScattering 
      //      prob_ss+prob_sl+prob_bs:1            -> LambertianReflection
      //
	void DoAbsorption();
	void DoReflection();

private:
    CG4*          m_g4 ; 
    CMaterialLib* m_mlib ; 
    Opticks*      m_ok ; 
#ifdef SCB_REFLECT_CHEAT
    bool          m_reflectcheat ; 
#endif

#ifdef SCB_BND_DEBUG
    bool     m_dbg ; 
    bool     m_other ; 
    int      m_event_id ; 
    int      m_photon_id ; 
    int      m_step_id ; 
#endif

	G4double thePhotonMomentum;

	G4ThreeVector OldMomentum;
	G4ThreeVector OldPolarization;

	G4ThreeVector NewMomentum;
	G4ThreeVector NewPolarization;

	G4ThreeVector theGlobalNormal;
	G4ThreeVector theFacetNormal;

	G4Material* Material1;
	G4Material* Material2;

	G4OpticalSurface* OpticalSurface;

	G4double Rindex1;
	G4double Rindex2;

	G4double cost1, cost2, sint1, sint2;

	Ds::DsG4OpBoundaryProcessStatus theStatus;

	G4OpticalSurfaceModel theModel;

	G4OpticalSurfaceFinish theFinish;

	G4double theReflectivity;
	G4double theEfficiency;
	G4double prob_sl, prob_ss, prob_bs;

        G4int iTE, iTM;

        G4double kCarTolerance;

        G4int abNormalCounter;

        G4int m_DielectricMetal_LambertianReflection ; 
};

////////////////////
// Inline methods
////////////////////

inline
G4bool DsG4OpBoundaryProcess::G4BooleanRand(const G4double prob) 
{
  /* Returns a random boolean variable with the specified probability */

  return (CG4UniformRand(__FILE__,__LINE__) < prob);
}

inline
G4bool DsG4OpBoundaryProcess::IsApplicable(const G4ParticleDefinition& 
					               aParticleType)
{
   return ( &aParticleType == G4OpticalPhoton::OpticalPhoton() );
}

inline
G4OpticalSurfaceModel DsG4OpBoundaryProcess::GetModel() const
{
   return theModel;
}

inline
Ds::DsG4OpBoundaryProcessStatus DsG4OpBoundaryProcess::GetStatus() const
{
   return theStatus;
}

inline
void DsG4OpBoundaryProcess::SetModel(G4OpticalSurfaceModel model)
{
   theModel = model;
}

inline
void DsG4OpBoundaryProcess::ChooseReflection()
{
                 G4double rand = CG4UniformRand(__FILE__,__LINE__);
                 if ( rand >= 0.0 && rand < prob_ss ) {
                    theStatus = Ds::SpikeReflection;
                    theFacetNormal = theGlobalNormal;
                 }
                 else if ( rand >= prob_ss &&
                           rand <= prob_ss+prob_sl) {
                    theStatus = Ds::LobeReflection;
                 }
                 else if ( rand > prob_ss+prob_sl &&
                           rand < prob_ss+prob_sl+prob_bs ) {
                    theStatus = Ds::BackScattering;
                 }
                 else {
                    theStatus = Ds::LambertianReflection;
                 }
}


inline
void DsG4OpBoundaryProcess::DoReflection()
{
        if ( theStatus == Ds::LambertianReflection ) {

          NewMomentum = G4LambertianRand(theGlobalNormal);
          theFacetNormal = (NewMomentum - OldMomentum).unit();

        }
        else if ( theFinish == ground ) {

	  theStatus = Ds::LobeReflection;
          theFacetNormal = GetFacetNormal(OldMomentum,theGlobalNormal);
          G4double PdotN = OldMomentum * theFacetNormal;
          NewMomentum = OldMomentum - (2.*PdotN)*theFacetNormal;

        }
        else {

          theStatus = Ds::SpikeReflection;
          theFacetNormal = theGlobalNormal;
          G4double PdotN = OldMomentum * theFacetNormal;
          NewMomentum = OldMomentum - (2.*PdotN)*theFacetNormal;

        }
        G4double EdotN = OldPolarization * theFacetNormal;
        NewPolarization = -OldPolarization + (2.*EdotN)*theFacetNormal;
}

#endif /* DsG4OpBoundaryProcess_h */
