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
////////////////////////////////////////////////////////////////////////
// Optical Photon Boundary Process Class Implementation
////////////////////////////////////////////////////////////////////////
//
// File:        G4OpBoundaryProcess.cc
// Description: Discrete Process -- reflection/refraction at
//                                  optical interfaces
// Version:     1.1
// Created:     1997-06-18
// Modified:    1998-05-25 - Correct parallel component of polarization
//                           (thanks to: Stefano Magni + Giovanni Pieri)
//              1998-05-28 - NULL Rindex pointer before reuse
//                           (thanks to: Stefano Magni)
//              1998-06-11 - delete *sint1 in oblique reflection
//                           (thanks to: Giovanni Pieri)
//              1998-06-19 - move from GetLocalExitNormal() to the new 
//                           method: GetLocalExitNormal(&valid) to get
//                           the surface normal in all cases
//              1998-11-07 - NULL OpticalSurface pointer before use
//                           comparison not sharp for: std::abs(cost1) < 1.0
//                           remove sin1, sin2 in lines 556,567
//                           (thanks to Stefano Magni)
//              1999-10-10 - Accommodate changes done in DoAbsorption by
//                           changing logic in DielectricMetal
//              2001-10-18 - avoid Linux (gcc-2.95.2) warning about variables
//                           might be used uninitialized in this function
//                           moved E2_perp, E2_parl and E2_total out of 'if'
//              2003-11-27 - Modified line 168-9 to reflect changes made to
//                           G4OpticalSurface class ( by Fan Lei)
//              2004-02-02 - Set theStatus = Undefined at start of DoIt
//              2005-07-28 - add G4ProcessType to constructor
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
// 		adopted from work by Werner Keil - April 2/96
// mail:        gum@triumf.ca
//
////////////////////////////////////////////////////////////////////////

#include "G4ios.hh"
#include "G4PhysicalConstants.hh"
#include "G4OpProcessSubType.hh"

#include "InstrumentedG4OpBoundaryProcess.hh"
#include "G4GeometryTolerance.hh"

#include "G4VSensitiveDetector.hh"
#include "G4ParallelWorldProcess.hh"

#include "G4SystemOfUnits.hh"


#include <csignal>

#include "JPMT.h"
#include "Layr.h"

#include "SLOG.hh"
#include "spho.h"
#include "STrackInfo.h"
#include "U4OpticalSurface.h"
#include "U4MaterialPropertiesTable.h"

#include "U4UniformRand.h"
NP* U4UniformRand::UU = nullptr ; 

#ifdef DEBUG_PIDX

#include "scuda.h"
#include "squad.h"
#include "SEvt.hh"
#include "SSys.hh"
#include "U4PhotonInfo.h"
const int InstrumentedG4OpBoundaryProcess::PIDX = SSys::getenvint("PIDX", -1) ; 
#endif


#ifdef DEBUG_TAG
#include "U4Stack.h"
#include "SEvt.hh"



const plog::Severity InstrumentedG4OpBoundaryProcess::LEVEL = SLOG::EnvLevel("InstrumentedG4OpBoundaryProcess", "DEBUG" ); 

const bool InstrumentedG4OpBoundaryProcess::FLOAT  = getenv("InstrumentedG4OpBoundaryProcess_FLOAT") != nullptr ;

//inline void InstrumentedG4OpBoundaryProcess::ResetNumberOfInteractionLengthLeft(){ G4VProcess::ResetNumberOfInteractionLengthLeft(); }
inline void InstrumentedG4OpBoundaryProcess::ResetNumberOfInteractionLengthLeft()
{
    //std::cout << "InstrumentedG4OpBoundaryProcess::FLOAT " << FLOAT << std::endl ; 
    G4double u = G4UniformRand() ; 

    LOG(LEVEL) << U4UniformRand::Desc(u) ; 


    SEvt::AddTag( U4Stack_BoundaryDiscreteReset, u );  

    if(FLOAT)
    {   
        float f = -1.f*std::log( float(u) ) ;   
        theNumberOfInteractionLengthLeft = f ; 
    }   
    else
    {   
        theNumberOfInteractionLengthLeft = -1.*G4Log(u) ;   
    }   
    theInitialNumberOfInteractionLength = theNumberOfInteractionLengthLeft; 

}



#endif







/////////////////////////
// Class Implementation
/////////////////////////

        //////////////
        // Operators
        //////////////

// InstrumentedG4OpBoundaryProcess::operator=(const InstrumentedG4OpBoundaryProcess &right)
// {
// }

        /////////////////
        // Constructors
        /////////////////

InstrumentedG4OpBoundaryProcess::InstrumentedG4OpBoundaryProcess(const G4String& processName,
                                               G4ProcessType type)
             : G4VDiscreteProcess(processName, type)
{
        PostStepDoIt_count = -1 ; 

        if ( verboseLevel > 0) {
           G4cout << GetProcessName() << " is created " << G4endl;
        }

        LOG(LEVEL) << " processName " << GetProcessName()  ; 

        SetProcessSubType(fOpBoundary);

        jpmt = new JPMT ; 

        theStatus = Undefined;
        theModel = glisur;
        theFinish = polished;
        theReflectivity =  1.;
        theEfficiency   =  0.;
        theTransmittance = 0.;

        theSurfaceRoughness = 0.;

        prob_sl = 0.;
        prob_ss = 0.;
        prob_bs = 0.;

        PropertyPointer  = NULL;
        PropertyPointer1 = NULL;
        PropertyPointer2 = NULL;

        Material1 = NULL;
        Material2 = NULL;

        OpticalSurface = NULL;

        kCarTolerance = G4GeometryTolerance::GetInstance()
                        ->GetSurfaceTolerance();

        iTE = iTM = 0;
        thePhotonMomentum = 0.;
        Rindex1 = Rindex2 = 1.;
        cost1 = cost2 = sint1 = sint2 = 0.;

        idx = idy = 0;
        DichroicVector = NULL;

        fInvokeSD = true;
}

// InstrumentedG4OpBoundaryProcess::InstrumentedG4OpBoundaryProcess(const InstrumentedG4OpBoundaryProcess &right)
// {
// }

        ////////////////
        // Destructors
        ////////////////

InstrumentedG4OpBoundaryProcess::~InstrumentedG4OpBoundaryProcess(){}

        ////////////
        //// Methods
        ////////////

// PostStepDoIt
// ------------
//

G4VParticleChange*
InstrumentedG4OpBoundaryProcess::PostStepDoIt(const G4Track& aTrack, const G4Step& aStep)
{
    PostStepDoIt_count += 1 ;  
    spho* label = STrackInfo<spho>::GetRef(&aTrack) ; 

    LOG(LEVEL) << "[ " <<  PostStepDoIt_count  ;   
    LOG(LEVEL) 
        << " PostStepDoIt_count " << PostStepDoIt_count
        << " label.desc " << label->desc()
        ; 

    // coping with the spagetti mush  
    G4VParticleChange* change = PostStepDoIt_(aTrack, aStep) ; 

    LOG(LEVEL) << "] " << PostStepDoIt_count  ; 
    return change ; 
}

G4VParticleChange*
InstrumentedG4OpBoundaryProcess::PostStepDoIt_(const G4Track& aTrack, const G4Step& aStep)
{
#ifdef DEBUG_PIDX
        // U4PhotonInfo::GetIndex is picking up the index from the label set 
        // in U4Recorder::PreUserTrackingAction_Optical for initially unlabelled input photons
        pidx = U4PhotonInfo::GetIndex(&aTrack);   
        pidx_dump = pidx == PIDX ; 
        // HUH: observed this happening twice for each pidx with what looks like same step ?
       /*
        std::cout 
            << "InstrumentedG4OpBoundaryProcess::PostStepDoIt"
            << " pidx " << std::setw(4) << pidx
            << " PIDX " << std::setw(4) << PIDX
            << " pidx_dump " << std::setw(4) << pidx_dump
            << " aStep " << &aStep
            << std::endl 
            ;
        */
#endif
        theStatus = Undefined;

        aParticleChange.Initialize(aTrack);
        aParticleChange.ProposeVelocity(aTrack.GetVelocity());

        // Get hyperStep from  G4ParallelWorldProcess
        //  NOTE: PostSetpDoIt of this process should be
        //        invoked after G4ParallelWorldProcess!

        const G4Step* pStep = &aStep;

        const G4Step* hStep = G4ParallelWorldProcess::GetHyperStep();
        
        if (hStep) pStep = hStep;

        G4bool isOnBoundary =
                (pStep->GetPostStepPoint()->GetStepStatus() == fGeomBoundary);

        if (isOnBoundary) {
           Material1 = pStep->GetPreStepPoint()->GetMaterial();
           Material2 = pStep->GetPostStepPoint()->GetMaterial();
        } else {
           theStatus = NotAtBoundary;
           if ( verboseLevel > 0) BoundaryProcessVerbose();
           return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
        }

        G4VPhysicalVolume* thePrePV  =
                               pStep->GetPreStepPoint() ->GetPhysicalVolume();
        G4VPhysicalVolume* thePostPV =
                               pStep->GetPostStepPoint()->GetPhysicalVolume();

        LOG(LEVEL)
            << " PostStepDoIt_count " << PostStepDoIt_count
            << " thePrePV " << ( thePrePV ? thePrePV->GetName() : "-" )
            << " thePostPV " << ( thePostPV ? thePostPV->GetName() : "-" )
            ;

        if ( verboseLevel > 0 ) {
           G4cout << " Photon at Boundary! " << G4endl;
           if (thePrePV)  G4cout << " thePrePV:  " << thePrePV->GetName()  << G4endl;
           if (thePostPV) G4cout << " thePostPV: " << thePostPV->GetName() << G4endl;
        }

        if (aTrack.GetStepLength()<=kCarTolerance/2){
                theStatus = StepTooSmall;
                if ( verboseLevel > 0) BoundaryProcessVerbose();
                return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
        }

        const G4DynamicParticle* aParticle = aTrack.GetDynamicParticle();

        thePhotonMomentum = aParticle->GetTotalMomentum();
        OldMomentum       = aParticle->GetMomentumDirection();
        OldPolarization   = aParticle->GetPolarization();

        if ( verboseLevel > 0 ) {
           G4cout << " Old Momentum Direction: " << OldMomentum     << G4endl;
           G4cout << " Old Polarization:       " << OldPolarization << G4endl;
        }

        theGlobalPoint = pStep->GetPostStepPoint()->GetPosition();

        G4bool valid;
        //  Use the new method for Exit Normal in global coordinates,
        //    which provides the normal more reliably.

        // ID of Navigator which limits step

        G4int hNavId = G4ParallelWorldProcess::GetHypNavigatorID();
        std::vector<G4Navigator*>::iterator iNav =
                G4TransportationManager::GetTransportationManager()->
                                         GetActiveNavigatorsIterator();
        theGlobalNormal =
                   (iNav[hNavId])->GetGlobalExitNormal(theGlobalPoint,&valid);
#ifdef DEBUG_PIDX
        {
            quad2& prd = SEvt::Get()->current_prd ; 
            prd.q0.f.x = theGlobalNormal.x() ; 
            prd.q0.f.y = theGlobalNormal.y() ; 
            prd.q0.f.z = theGlobalNormal.z() ; 

            // TRY USING PRE->POST POSITION CHANGE TO GET THE PRD DISTANCE ? 
            G4ThreeVector theGlobalPoint_pre = pStep->GetPreStepPoint()->GetPosition();
            G4ThreeVector theGlobalPoint_delta = theGlobalPoint - theGlobalPoint_pre  ;  
            prd.q0.f.w = theGlobalPoint_delta.mag() ; 

           // HMM: PRD intersect identity ? how to mimic what Opticks does ? 
       }
#endif

        if (valid) {
          theGlobalNormal = -theGlobalNormal;
        }
        else 
        {
          G4ExceptionDescription ed;
          ed << " InstrumentedG4OpBoundaryProcess/PostStepDoIt(): "
                 << " The Navigator reports that it returned an invalid normal"
                 << G4endl;
          G4Exception("InstrumentedG4OpBoundaryProcess::PostStepDoIt", "OpBoun01",
                      EventMustBeAborted,ed,
                      "Invalid Surface Normal - Geometry must return valid surface normal");
        }

        if (OldMomentum * theGlobalNormal > 0.0) {
#ifdef G4OPTICAL_DEBUG
           G4ExceptionDescription ed;
           ed << " InstrumentedG4OpBoundaryProcess/PostStepDoIt(): "
              << " theGlobalNormal points in a wrong direction. "
              << G4endl;
           ed << "    The momentum of the photon arriving at interface (oldMomentum)"
              << " must exit the volume cross in the step. " << G4endl;
           ed << "  So it MUST have dot < 0 with the normal that Exits the new volume (globalNormal)." << G4endl;
           ed << "  >> The dot product of oldMomentum and global Normal is " << OldMomentum*theGlobalNormal << G4endl;
           ed << "     Old Momentum  (during step)     = " << OldMomentum << G4endl;
           ed << "     Global Normal (Exiting New Vol) = " << theGlobalNormal << G4endl;
           ed << G4endl;
           G4Exception("InstrumentedG4OpBoundaryProcess::PostStepDoIt", "OpBoun02",
                       EventMustBeAborted,  // Or JustWarning to see if it happens repeatedbly on one ray
                       ed,
                      "Invalid Surface Normal - Geometry must return valid surface normal pointing in the right direction");
#else
           theGlobalNormal = -theGlobalNormal;
#endif
        }

	G4MaterialPropertiesTable* aMaterialPropertiesTable;
        G4MaterialPropertyVector* Rindex;

	aMaterialPropertiesTable = Material1->GetMaterialPropertiesTable();
        if (aMaterialPropertiesTable) {
		Rindex = aMaterialPropertiesTable->GetProperty(kRINDEX);
	}
	else {
                theStatus = NoRINDEX;
                if ( verboseLevel > 0) BoundaryProcessVerbose();
                aParticleChange.ProposeLocalEnergyDeposit(thePhotonMomentum);
                aParticleChange.ProposeTrackStatus(fStopAndKill);
                return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
	}

        if (Rindex) {
           Rindex1 = Rindex->Value(thePhotonMomentum);
        }
        else {
	        theStatus = NoRINDEX;
                if ( verboseLevel > 0) BoundaryProcessVerbose();
                aParticleChange.ProposeLocalEnergyDeposit(thePhotonMomentum);
                aParticleChange.ProposeTrackStatus(fStopAndKill);
                return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
	}


        theReflectivity =  1.;
        theEfficiency   =  0.;
        theTransmittance = 0.;

        theSurfaceRoughness = 0.;

        theModel = glisur;
        theFinish = polished;

        G4SurfaceType type = dielectric_dielectric;

        Rindex = NULL;
        OpticalSurface = NULL;

        G4LogicalSurface* Surface = NULL;

        Surface = G4LogicalBorderSurface::GetSurface(thePrePV, thePostPV);




//#ifdef DEBUG_PIDX
        //std::cout << "InstrumentedG4OpBoundaryProcess::PostStepDoIt Surface " << ( Surface ? "Y" : "N" ) << std::endl ; 
        //if(Surface) std::raise(SIGINT) ; 
//#endif


        if (Surface == NULL){
          G4bool enteredDaughter= (thePostPV->GetMotherLogical() ==
                                   thePrePV ->GetLogicalVolume());
	  if(enteredDaughter){
	    Surface = 
              G4LogicalSkinSurface::GetSurface(thePostPV->GetLogicalVolume());
	    if(Surface == NULL)
	      Surface =
                G4LogicalSkinSurface::GetSurface(thePrePV->GetLogicalVolume());
	  }
	  else {
	    Surface =
              G4LogicalSkinSurface::GetSurface(thePrePV->GetLogicalVolume());
	    if(Surface == NULL)
	      Surface =
                G4LogicalSkinSurface::GetSurface(thePostPV->GetLogicalVolume());
	  }
	}



        if (Surface) OpticalSurface = 
           dynamic_cast <G4OpticalSurface*> (Surface->GetSurfaceProperty());


        char opsu =  OpticalSurface ? OpticalSurface->GetName().c_str()[0] : '_' ;  
        bool custom = opsu == '@' ; 

    LOG(LEVEL) 
        << " PostStepDoIt_count " << PostStepDoIt_count
        << " Surface " << ( Surface ? Surface->GetName() : "-" ) 
        << " OpticalSurface " << ( OpticalSurface ? OpticalSurface->GetName() : "-" ) 
        << " opsu " << opsu 
        << " custom " << custom 
        ; 


        if (OpticalSurface) {


           LOG(LEVEL) 
               << " PostStepDoIt_count " << PostStepDoIt_count << " "
               << U4OpticalSurface::Desc(OpticalSurface) 
               ; 


           type      = OpticalSurface->GetType();
           theModel  = OpticalSurface->GetModel();
           theFinish = OpticalSurface->GetFinish();

           aMaterialPropertiesTable = OpticalSurface->
                                        GetMaterialPropertiesTable();


           if (aMaterialPropertiesTable) {

               LOG(LEVEL) 
                   << " PostStepDoIt_count " << PostStepDoIt_count << " " 
                   << U4MaterialPropertiesTable::Desc(aMaterialPropertiesTable) 
                   ; 

              if (theFinish == polishedbackpainted || theFinish == groundbackpainted ) 
              {
                  Rindex = aMaterialPropertiesTable->GetProperty(kRINDEX);
	              if (Rindex) 
                  {
                      Rindex2 = Rindex->Value(thePhotonMomentum);
                  }
                  else 
                  {
                     theStatus = NoRINDEX;
                     if ( verboseLevel > 0) BoundaryProcessVerbose();
                     aParticleChange.ProposeLocalEnergyDeposit(thePhotonMomentum);
                     aParticleChange.ProposeTrackStatus(fStopAndKill);
                     return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
                  }
              }

              PropertyPointer =
                      aMaterialPropertiesTable->GetProperty(kREFLECTIVITY);
              PropertyPointer1 =
                      aMaterialPropertiesTable->GetProperty(kREALRINDEX);
              PropertyPointer2 =
                      aMaterialPropertiesTable->GetProperty(kIMAGINARYRINDEX);


              LOG(LEVEL) 
                   << " PostStepDoIt_count " << PostStepDoIt_count
                   << " PropertyPointer.kREFLECTIVITY " << PropertyPointer
                   << " PropertyPointer1.kREALRINDEX " << PropertyPointer1 
                   << " PropertyPointer2.kIMAGINARYRINDEX " << PropertyPointer2
                   ;

              iTE = 1;
              iTM = 1;



              if(custom) CustomART(aTrack, aStep) ;  // FOR NOW DONT CHANGE RESULTS 

              if (PropertyPointer) 
              {
                 theReflectivity =
                          PropertyPointer->Value(thePhotonMomentum);

              } 
              else if (PropertyPointer1 && PropertyPointer2) 
              {
                 CalculateReflectivity();
              }

              PropertyPointer =
              aMaterialPropertiesTable->GetProperty(kEFFICIENCY);
              if (PropertyPointer) {
                      theEfficiency =
                      PropertyPointer->Value(thePhotonMomentum);
              }

              PropertyPointer =
              aMaterialPropertiesTable->GetProperty(kTRANSMITTANCE);
              if (PropertyPointer) {
                      theTransmittance =
                      PropertyPointer->Value(thePhotonMomentum);
              }


              LOG(LEVEL)
                  << " PostStepDoIt_count " << PostStepDoIt_count
                  << " theReflectivity " << theReflectivity
                  << " theEfficiency " << theEfficiency
                  << " theTransmittance " << theTransmittance
                  ; 


              if (aMaterialPropertiesTable->
                                     ConstPropertyExists("SURFACEROUGHNESS"))
                 theSurfaceRoughness = aMaterialPropertiesTable->
                                         GetConstProperty(kSURFACEROUGHNESS);

	      if ( theModel == unified ) {
                 PropertyPointer =
                 aMaterialPropertiesTable->GetProperty(kSPECULARLOBECONSTANT);
                 if (PropertyPointer) {
                         prob_sl =
                         PropertyPointer->Value(thePhotonMomentum);
                 } else {
                         prob_sl = 0.0;
                 }

                 PropertyPointer =
                 aMaterialPropertiesTable->GetProperty(kSPECULARSPIKECONSTANT);
	         if (PropertyPointer) {
                         prob_ss =
                         PropertyPointer->Value(thePhotonMomentum);
                 } else {
                         prob_ss = 0.0;
                 }

                 PropertyPointer =
                 aMaterialPropertiesTable->GetProperty(kBACKSCATTERCONSTANT);
                 if (PropertyPointer) {
                         prob_bs =
                         PropertyPointer->Value(thePhotonMomentum);
                 } else {
                         prob_bs = 0.0;
                 }
              }
           }
           else if (theFinish == polishedbackpainted ||
                    theFinish == groundbackpainted ) {
                      aParticleChange.ProposeLocalEnergyDeposit(thePhotonMomentum);
                      aParticleChange.ProposeTrackStatus(fStopAndKill);
                      return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
           }
        }
  
        LOG(LEVEL) 
            << " PostStepDoIt_count " << PostStepDoIt_count
            << " after OpticalSurface if "  
            ;

        if (type == dielectric_dielectric ) {
           if (theFinish == polished || theFinish == ground ) {

              if (Material1 == Material2){
                 theStatus = SameMaterial;
                 if ( verboseLevel > 0) BoundaryProcessVerbose();
		 return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
	      }
              aMaterialPropertiesTable =
                     Material2->GetMaterialPropertiesTable();
              if (aMaterialPropertiesTable)
                 Rindex = aMaterialPropertiesTable->GetProperty(kRINDEX);
              if (Rindex) {
                 Rindex2 = Rindex->Value(thePhotonMomentum);
              }
              else {
                 theStatus = NoRINDEX;
                 if ( verboseLevel > 0) BoundaryProcessVerbose();
                 aParticleChange.ProposeLocalEnergyDeposit(thePhotonMomentum);
                 aParticleChange.ProposeTrackStatus(fStopAndKill);
                 return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
              }
           }

           LOG(LEVEL) 
               << " PostStepDoIt_count " << PostStepDoIt_count
               << " dielectric_dielectric " 
               << " Rindex2 " << Rindex2 
               ;  
        }

	if (type == dielectric_metal) {
     
          LOG(LEVEL) 
              << " PostStepDoIt_count " << PostStepDoIt_count
              << " dielectric_metal " 
              ;  
 
          DielectricMetal();

	}
        else if (type == dielectric_LUT) {

          DielectricLUT();

        }
        else if (type == dielectric_LUTDAVIS) {

          DielectricLUTDAVIS();

        }
        else if (type == dielectric_dichroic) {

          DielectricDichroic();

        }
        else if (type == dielectric_dielectric) {

          if ( theFinish == polishedbackpainted ||
               theFinish == groundbackpainted ) {
             DielectricDielectric();
          }
          else {



             G4double rand = G4UniformRand();
             LOG(LEVEL) 
                 << " PostStepDoIt_count " << PostStepDoIt_count
                 << " didi.rand " << U4UniformRand::Desc(rand) 
                 << " theReflectivity " << std::setw(10) << std::fixed << std::setprecision(4) << theReflectivity
                 << " theTransmittance " << std::setw(10) << std::fixed << std::setprecision(4) << theTransmittance
                 ; 

#ifdef DEBUG_TAG
             SEvt::AddTag(U4Stack_BoundaryBurn_SurfaceReflectTransmitAbsorb, rand );  
#endif

#ifdef DEBUG_PIDX
             // SCB theReflectivity default is 1. so  "rand > theReflectivity" always false
             //     meaning that "rand" is always a burn doing nothing.  
             //     Unless a surface is associated which changes theReflectivity to be less than 1. 
             //     which can make the "rand" actually control the Reflect-OR-Absorb-OR-Transmit decision.  
             //
             if(pidx_dump)
             {
                  std::cout 
                      << " DiDi0 " 
                      << " pidx " << std::setw(6) << pidx 
                      << " rand " << std::setw(10) << std::fixed << std::setprecision(5) << rand  
                      << " theReflectivity " << std::setw(10) << std::fixed << std::setprecision(4) << theReflectivity
                      << " rand > theReflectivity  " << (rand > theReflectivity )
                      << std::endl 
                      ;
             }
#endif
             if ( rand > theReflectivity ) {   
                if (rand > theReflectivity + theTransmittance) { 
                   DoAbsorption();
                } else {
                   theStatus = Transmission;
                   NewMomentum = OldMomentum;
                   NewPolarization = OldPolarization;
                }
             }
             else {
                if ( theFinish == polishedfrontpainted ) {
                   DoReflection();
                }
                else if ( theFinish == groundfrontpainted ) {
                   theStatus = LambertianReflection;
                   DoReflection();
                }
                else {
                   DielectricDielectric();
                }
             }
          }
        }
        else {

          G4cerr << " Error: G4BoundaryProcess: illegal boundary type " << G4endl;
          return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);

        }

        NewMomentum = NewMomentum.unit();
        NewPolarization = NewPolarization.unit();

        if ( verboseLevel > 0) {
           G4cout << " New Momentum Direction: " << NewMomentum     << G4endl;
           G4cout << " New Polarization:       " << NewPolarization << G4endl;
           BoundaryProcessVerbose();
        }

        aParticleChange.ProposeMomentumDirection(NewMomentum);
        aParticleChange.ProposePolarization(NewPolarization);

        if ( theStatus == FresnelRefraction || theStatus == Transmission ) {
           G4MaterialPropertyVector* groupvel =
           Material2->GetMaterialPropertiesTable()->GetProperty(kGROUPVEL);
           G4double finalVelocity = groupvel->Value(thePhotonMomentum);
           aParticleChange.ProposeVelocity(finalVelocity);
        }

        if ( theStatus == Detection && fInvokeSD ) InvokeSD(pStep);

        return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
}

void InstrumentedG4OpBoundaryProcess::CustomART(const G4Track& aTrack, const G4Step& aStep )
{
    G4double energy = thePhotonMomentum ; 
    G4double wavelength = twopi*hbarc/energy ;

    G4double energy_eV = energy/eV ;
    G4double wavelength_nm = wavelength/nm ; 

    const G4AffineTransform& transform = aTrack.GetTouchable()->GetHistory()->GetTopTransform();

    G4ThreeVector localPoint        = transform.TransformPoint(theGlobalPoint);
    G4ThreeVector localMomentum     = transform.TransformAxis(OldMomentum);
    G4ThreeVector localPolarization = transform.TransformAxis(OldPolarization);
    G4ThreeVector localNormal       = transform.TransformAxis(theGlobalNormal);

    G4double minus_cos_theta_global = OldMomentum*theGlobalNormal ;

    const G4ThreeVector& surface_normal = localNormal ; 
    const G4ThreeVector& direction = localMomentum ; 
    const G4ThreeVector& polarization = localPolarization ; 

    G4double minus_cos_theta = direction*surface_normal  ; 
    G4ThreeVector oriented_normal = ( minus_cos_theta < 0. ? 1. : -1. )*surface_normal ;


    LOG(LEVEL) 
        << std::endl
        << " PostStepDoIt_count " << PostStepDoIt_count  
        << std::endl
        << " energy_eV " << energy_eV 
        << std::endl
        << " wavelength_nm " << wavelength_nm
        << std::endl
        << " OldMomentum " << OldMomentum
        << std::endl
        << " OldPolarization " << OldPolarization
        << std::endl
        << " theGlobalPoint " << theGlobalPoint
        << std::endl
        << " theGlobalNormal " << theGlobalNormal
        << std::endl
        << " transform " << transform
        << std::endl
        << " localPoint " << localPoint
        << std::endl
        << " localMomentum " << localMomentum
        << std::endl
        << " localPolarization " << localPolarization
        << std::endl
        << " localNormal " << localNormal << " TODO: check rigidly outwards for backwards photons " 
        << std::endl
        << " minus_cos_theta_global " << minus_cos_theta_global
        << std::endl
        << " minus_cos_theta " << minus_cos_theta << " local and global should make no difference " 
        ;


    int pmtcat = JPMT::HAMA ;  // TODO: pmtcat ctor argument ? HMM: NNVT_HiQE ?
    //double _qe = 0.5 ; 
    double _qe = 0.0 ; 

    StackSpec<double> spec ; 
    spec.d0  = 0. ; 
    spec.d1  = jpmt->get_thickness_nm( pmtcat, JPMT::L1 );  
    spec.d2  = jpmt->get_thickness_nm( pmtcat, JPMT::L2 );  
    spec.d3 = 0. ; 

    spec.n0r = jpmt->get_rindex( pmtcat, JPMT::L0, JPMT::RINDEX, energy_eV );  
    spec.n0i = jpmt->get_rindex( pmtcat, JPMT::L0, JPMT::KINDEX, energy_eV );

    spec.n1r = jpmt->get_rindex( pmtcat, JPMT::L1, JPMT::RINDEX, energy_eV );
    spec.n1i = jpmt->get_rindex( pmtcat, JPMT::L1, JPMT::KINDEX, energy_eV );

    spec.n2r = jpmt->get_rindex( pmtcat, JPMT::L2, JPMT::RINDEX, energy_eV );  
    spec.n2i = jpmt->get_rindex( pmtcat, JPMT::L2, JPMT::KINDEX, energy_eV );  

    spec.n3r = jpmt->get_rindex( pmtcat, JPMT::L3, JPMT::RINDEX, energy_eV );  
    spec.n3i = jpmt->get_rindex( pmtcat, JPMT::L3, JPMT::KINDEX, energy_eV );

    Stack<double,4> stack(      wavelength_nm, minus_cos_theta, spec );  // NB stack is flipped for minus_cos_theta > 0. 
    Stack<double,4> stackNormal(wavelength_nm, -1.            , spec );  // minus_cos_theta -1. means normal incidence and stack not flipped

    // NB stack is flipped for minus_cos_theta > 0. so:
    //
    //    stack.ll[0] always incident side
    //    stack.ll[3] always transmission side 
    //
    // stackNormal is not flipped, presumably due to _qe definition


    double _n0         = stack.ll[0].n.real() ; 
    double _sin_theta0 = stack.ll[0].st.real() ; 
    double _cos_theta0 = stack.ll[0].ct.real() ;

    double _n3         = stack.ll[3].n.real() ; 
    double _cos_theta3 = stack.ll[3].ct.real() ;


 
    double E_s2 = _sin_theta0 > 0. ? (polarization*direction.cross(oriented_normal))/_sin_theta0 : 0. ; 
    E_s2 *= E_s2;

    double fT_s = stack.art.T_s ; 
    double fT_p = stack.art.T_p ; 
    double fR_s = stack.art.R_s ; 
    double fR_p = stack.art.R_p ; 

    double fT_n = stackNormal.art.T ; 
    double fR_n = stackNormal.art.R ; 

    double one = 1.0 ; 
    double T = fT_s*E_s2 + fT_p*(one-E_s2);
    double R = fR_s*E_s2 + fR_p*(one-E_s2);
    double A = one - (T+R);
    double An = one - (fT_n+fR_n);
    double detect_fraction = _qe/An;

    LOG_IF(error, detect_fraction > 1.)
         << " detect_fraction > 1. : " << detect_fraction
         << " _qe " << _qe
         << " An " << An
         ;

    double u0 = G4UniformRand();
    double u1 = G4UniformRand();

    char status = '?' ;
    if(     u0 < A)    status = u1 < detect_fraction ? 'D' : 'A' ;
    else if(u0 < A+R)  status = 'R' ;
    else               status = 'T' ;

    // HMM: could the standard methods be used instead of this decision ?


    G4ThreeVector new_direction(direction); 
    G4ThreeVector new_polarization(polarization); 

    // the below is copying junoPMTOpticalModel (TODO: need to compare with G4OpBoundaryProcess)
    if( status == 'R' )
    {
        new_direction    -= 2.*(new_direction*oriented_normal)*oriented_normal ;
        new_polarization -= 2.*(new_polarization*oriented_normal)*oriented_normal ;
    }
    else if( status == 'T' )
    {
        new_direction = (_cos_theta3 - _cos_theta0*_n0/_n3)*oriented_normal + (_n0/_n3)*new_direction;
        // not normalized ?
        new_polarization = (new_polarization-(new_polarization*direction)*direction).unit();
    }

    LOG(LEVEL) 
        << " status " << status 
        << std::endl 
        << " new_direction " << new_direction 
        << std::endl 
        << " new_polarization " << new_polarization 
        ; 


    const G4Track* track = &aTrack ; 

    spho* label = STrackInfo<spho>::GetRef(track);
    LOG_IF(fatal, !label)
        << " all photon tracks must be labelled "
        << " track " << &track
        << std::endl
        << STrackInfo<spho>::Desc(track)
        ;

    assert( label );
    label->uc4.w = status ;

    LOG(LEVEL)
        << " T " << std::setw(10) << std::fixed << std::setprecision(4) << T
        << " R " << std::setw(10) << std::fixed << std::setprecision(4) << R
        << " A " << std::setw(10) << std::fixed << std::setprecision(4) << A
        << " u0 " << std::setw(10) << std::fixed << std::setprecision(4) << u0
        << " u1 " << std::setw(10) << std::fixed << std::setprecision(4) << u1
        << " status " << status
        ;


     // cannot use DoAbsorption ? as that does theEfficieny random throw
    if(status == 'A' || status == 'D')
    {
        theStatus = status == 'D' ? Detection : Absorption  ;
        aParticleChange.ProposeLocalEnergyDeposit(status == 'D' ? thePhotonMomentum : 0.0);

        NewMomentum = OldMomentum;
        NewPolarization = OldPolarization;   // follow DoAbsorption
        aParticleChange.ProposeTrackStatus(fStopAndKill);
    }



}


void InstrumentedG4OpBoundaryProcess::BoundaryProcessVerbose() const
{
        if ( theStatus == Undefined )
                G4cout << " *** Undefined *** " << G4endl;
        if ( theStatus == Transmission )
                G4cout << " *** Transmission *** " << G4endl;
        if ( theStatus == FresnelRefraction )
                G4cout << " *** FresnelRefraction *** " << G4endl;
        if ( theStatus == FresnelReflection )
                G4cout << " *** FresnelReflection *** " << G4endl;
        if ( theStatus == TotalInternalReflection )
                G4cout << " *** TotalInternalReflection *** " << G4endl;
        if ( theStatus == LambertianReflection )
                G4cout << " *** LambertianReflection *** " << G4endl;
        if ( theStatus == LobeReflection )
                G4cout << " *** LobeReflection *** " << G4endl;
        if ( theStatus == SpikeReflection )
                G4cout << " *** SpikeReflection *** " << G4endl;
        if ( theStatus == BackScattering )
                G4cout << " *** BackScattering *** " << G4endl;
        if ( theStatus == PolishedLumirrorAirReflection )
                G4cout << " *** PolishedLumirrorAirReflection *** " << G4endl;
        if ( theStatus == PolishedLumirrorGlueReflection )
                G4cout << " *** PolishedLumirrorGlueReflection *** " << G4endl;
        if ( theStatus == PolishedAirReflection )
                G4cout << " *** PolishedAirReflection *** " << G4endl;
        if ( theStatus == PolishedTeflonAirReflection )
                G4cout << " *** PolishedTeflonAirReflection *** " << G4endl;
        if ( theStatus == PolishedTiOAirReflection )
                G4cout << " *** PolishedTiOAirReflection *** " << G4endl;
        if ( theStatus == PolishedTyvekAirReflection )
                G4cout << " *** PolishedTyvekAirReflection *** " << G4endl;
        if ( theStatus == PolishedVM2000AirReflection )
                G4cout << " *** PolishedVM2000AirReflection *** " << G4endl;
        if ( theStatus == PolishedVM2000GlueReflection )
                G4cout << " *** PolishedVM2000GlueReflection *** " << G4endl;
        if ( theStatus == EtchedLumirrorAirReflection )
                G4cout << " *** EtchedLumirrorAirReflection *** " << G4endl;
        if ( theStatus == EtchedLumirrorGlueReflection )
                G4cout << " *** EtchedLumirrorGlueReflection *** " << G4endl;
        if ( theStatus == EtchedAirReflection )
                G4cout << " *** EtchedAirReflection *** " << G4endl;
        if ( theStatus == EtchedTeflonAirReflection )
                G4cout << " *** EtchedTeflonAirReflection *** " << G4endl;
        if ( theStatus == EtchedTiOAirReflection )
                G4cout << " *** EtchedTiOAirReflection *** " << G4endl;
        if ( theStatus == EtchedTyvekAirReflection )
                G4cout << " *** EtchedTyvekAirReflection *** " << G4endl;
        if ( theStatus == EtchedVM2000AirReflection )
                G4cout << " *** EtchedVM2000AirReflection *** " << G4endl;
        if ( theStatus == EtchedVM2000GlueReflection )
                G4cout << " *** EtchedVM2000GlueReflection *** " << G4endl;
        if ( theStatus == GroundLumirrorAirReflection )
                G4cout << " *** GroundLumirrorAirReflection *** " << G4endl;
        if ( theStatus == GroundLumirrorGlueReflection )
                G4cout << " *** GroundLumirrorGlueReflection *** " << G4endl;
        if ( theStatus == GroundAirReflection )
                G4cout << " *** GroundAirReflection *** " << G4endl;
        if ( theStatus == GroundTeflonAirReflection )
                G4cout << " *** GroundTeflonAirReflection *** " << G4endl;
        if ( theStatus == GroundTiOAirReflection )
                G4cout << " *** GroundTiOAirReflection *** " << G4endl;
        if ( theStatus == GroundTyvekAirReflection )
                G4cout << " *** GroundTyvekAirReflection *** " << G4endl;
        if ( theStatus == GroundVM2000AirReflection )
                G4cout << " *** GroundVM2000AirReflection *** " << G4endl;
        if ( theStatus == GroundVM2000GlueReflection )
                G4cout << " *** GroundVM2000GlueReflection *** " << G4endl;
        if ( theStatus == Absorption )
                G4cout << " *** Absorption *** " << G4endl;
        if ( theStatus == Detection )
                G4cout << " *** Detection *** " << G4endl;
        if ( theStatus == NotAtBoundary )
                G4cout << " *** NotAtBoundary *** " << G4endl;
        if ( theStatus == SameMaterial )
                G4cout << " *** SameMaterial *** " << G4endl;
        if ( theStatus == StepTooSmall )
                G4cout << " *** StepTooSmall *** " << G4endl;
        if ( theStatus == NoRINDEX )
                G4cout << " *** NoRINDEX *** " << G4endl;
        if ( theStatus == Dichroic )
                G4cout << " *** Dichroic Transmission *** " << G4endl;
}

G4ThreeVector
InstrumentedG4OpBoundaryProcess::GetFacetNormal(const G4ThreeVector& Momentum,
			            const G4ThreeVector&  Normal ) const
{
        G4ThreeVector FacetNormal;

        if (theModel == unified || theModel == LUT || theModel== DAVIS) {

           /* This function code alpha to a random value taken from the
           distribution p(alpha) = g(alpha; 0, sigma_alpha)*std::sin(alpha),
           for alpha > 0 and alpha < 90, where g(alpha; 0, sigma_alpha)
           is a gaussian distribution with mean 0 and standard deviation
           sigma_alpha.  */

           G4double alpha;

           G4double sigma_alpha = 0.0;
           if (OpticalSurface) sigma_alpha = OpticalSurface->GetSigmaAlpha();

           if (sigma_alpha == 0.0) return FacetNormal = Normal;

           G4double f_max = std::min(1.0,4.*sigma_alpha);

           G4double phi, SinAlpha, CosAlpha, SinPhi, CosPhi, unit_x, unit_y, unit_z;
           G4ThreeVector tmpNormal;

           do {
              do {
                 alpha = G4RandGauss::shoot(0.0,sigma_alpha);
                 // Loop checking, 13-Aug-2015, Peter Gumplinger
              } while (G4UniformRand()*f_max > std::sin(alpha) || alpha >= halfpi );

              phi = G4UniformRand()*twopi;

              SinAlpha = std::sin(alpha);
              CosAlpha = std::cos(alpha);
              SinPhi = std::sin(phi);
              CosPhi = std::cos(phi);

              unit_x = SinAlpha * CosPhi;
              unit_y = SinAlpha * SinPhi;
              unit_z = CosAlpha;

              FacetNormal.setX(unit_x);
              FacetNormal.setY(unit_y);
              FacetNormal.setZ(unit_z);

              tmpNormal = Normal;

              FacetNormal.rotateUz(tmpNormal);
              // Loop checking, 13-Aug-2015, Peter Gumplinger
           } while (Momentum * FacetNormal >= 0.0);
	}
        else {

           G4double  polish = 1.0;
           if (OpticalSurface) polish = OpticalSurface->GetPolish();

           if (polish < 1.0) {
              do {
                 G4ThreeVector smear;
                 do {
                    smear.setX(2.*G4UniformRand()-1.0);
                    smear.setY(2.*G4UniformRand()-1.0);
                    smear.setZ(2.*G4UniformRand()-1.0);
                    // Loop checking, 13-Aug-2015, Peter Gumplinger
                 } while (smear.mag()>1.0);
                 smear = (1.-polish) * smear;
                 FacetNormal = Normal + smear;
                 // Loop checking, 13-Aug-2015, Peter Gumplinger
              } while (Momentum * FacetNormal >= 0.0);
              FacetNormal = FacetNormal.unit();
           }
           else {
              FacetNormal = Normal;
           }
	}
        return FacetNormal;
}


/**
InstrumentedG4OpBoundaryProcess::DielectricMetal
--------------------------------------------------

Changes::

   theStatus
   NewMomentum
   NewPolarization

**/

void InstrumentedG4OpBoundaryProcess::DielectricMetal()
{
       LOG(LEVEL) 
           << " PostStepDoIt_count " << PostStepDoIt_count
           ;

        G4int n = 0;
        G4double rand, PdotN, EdotN;
        G4ThreeVector A_trans, A_paral;

        do {

           n++;

           rand = G4UniformRand();

           LOG(LEVEL) 
               << " PostStepDoIt_count " << PostStepDoIt_count
               << " do-while n " << n 
               << " rand " << U4UniformRand::Desc(rand) 
               << " theReflectivity " << theReflectivity
               << " theTransmittance " << theTransmittance
               ;

#ifdef DEBUG_TAG
           SEvt::AddTag(U4Stack_BoundaryDiMeReflectivity, rand); 
#endif


           if ( rand > theReflectivity && n == 1 ) {
              if (rand > theReflectivity + theTransmittance) {
                DoAbsorption();
              } else {
                theStatus = Transmission;
                NewMomentum = OldMomentum;
                NewPolarization = OldPolarization;
              }
              LOG(LEVEL) << " rand > theReflectivity && n == 1  break " ; 
              break;
           }
           else {

             if (PropertyPointer1 && PropertyPointer2) {
                if ( n > 1 ) {
                   CalculateReflectivity();
                   if ( !G4BooleanRand_theReflectivity(theReflectivity) ) {
                      DoAbsorption();
                      break;
                   }
                }
             }

             if ( theModel == glisur || theFinish == polished ) {

                DoReflection();

             } else {

                if ( n == 1 ) ChooseReflection();
                                                                                
                if ( theStatus == LambertianReflection ) {
                   DoReflection();
                }
                else if ( theStatus == BackScattering ) {
                   NewMomentum = -OldMomentum;
                   NewPolarization = -OldPolarization;
                }
                else {

                   if(theStatus==LobeReflection){
                     if ( PropertyPointer1 && PropertyPointer2 ){
                     } else {
                        theFacetNormal =
                            GetFacetNormal(OldMomentum,theGlobalNormal);
                     }
                   }

                   PdotN = OldMomentum * theFacetNormal;
                   NewMomentum = OldMomentum - (2.*PdotN)*theFacetNormal;
                   EdotN = OldPolarization * theFacetNormal;

                   if (sint1 > 0.0 ) {
                      A_trans = OldMomentum.cross(theFacetNormal);
                      A_trans = A_trans.unit();
                   } else {
                      A_trans  = OldPolarization;
                   }
                   A_paral   = NewMomentum.cross(A_trans);
                   A_paral   = A_paral.unit();

                   if(iTE>0&&iTM>0) {
                     NewPolarization = 
                           -OldPolarization + (2.*EdotN)*theFacetNormal;
                   } else if (iTE>0) {
                     NewPolarization = -A_trans;
                   } else if (iTM>0) {
                     NewPolarization = -A_paral;
                   }

                }

             }

             LOG(LEVEL) << " Old<-New ? UGLY " ; 

             OldMomentum = NewMomentum;
             OldPolarization = NewPolarization;

	   }

       LOG(LEVEL) << " while check " ; 

          // Loop checking, 13-Aug-2015, Peter Gumplinger
	} while (NewMomentum * theGlobalNormal < 0.0);

    LOG(LEVEL) << " after while " ; 
}

void InstrumentedG4OpBoundaryProcess::DielectricLUT()
{
        G4int thetaIndex, phiIndex;
        G4double AngularDistributionValue, thetaRad, phiRad, EdotN;
        G4ThreeVector PerpendicularVectorTheta, PerpendicularVectorPhi;

        theStatus = G4OpBoundaryProcessStatus(G4int(theFinish) + 
                           (G4int(NoRINDEX)-G4int(groundbackpainted)));

        G4int thetaIndexMax = OpticalSurface->GetThetaIndexMax();
        G4int phiIndexMax   = OpticalSurface->GetPhiIndexMax();

        G4double rand;

        do {
           rand = G4UniformRand();
           if ( rand > theReflectivity ) {
              if (rand > theReflectivity + theTransmittance) {
                 DoAbsorption();
              } else {
                 theStatus = Transmission;
                 NewMomentum = OldMomentum;
                 NewPolarization = OldPolarization;
              }
              break;
           }
           else {
              // Calculate Angle between Normal and Photon Momentum
              G4double anglePhotonToNormal = 
                                          OldMomentum.angle(-theGlobalNormal);
              // Round it to closest integer
              G4int angleIncident = G4int(std::floor(180/pi*anglePhotonToNormal+0.5));

              // Take random angles THETA and PHI, 
              // and see if below Probability - if not - Redo
              do {
                 thetaIndex = G4RandFlat::shootInt(thetaIndexMax-1);
                 phiIndex = G4RandFlat::shootInt(phiIndexMax-1);
                 // Find probability with the new indeces from LUT
                 AngularDistributionValue = OpticalSurface -> 
                   GetAngularDistributionValue(angleIncident,
                                               thetaIndex,
                                               phiIndex);
                // Loop checking, 13-Aug-2015, Peter Gumplinger
              } while ( !G4BooleanRand(AngularDistributionValue) );

              thetaRad = (-90 + 4*thetaIndex)*pi/180;
              phiRad = (-90 + 5*phiIndex)*pi/180;
              // Rotate Photon Momentum in Theta, then in Phi
              NewMomentum = -OldMomentum;

              PerpendicularVectorTheta = NewMomentum.cross(theGlobalNormal);
              if (PerpendicularVectorTheta.mag() < kCarTolerance )
                          PerpendicularVectorTheta = NewMomentum.orthogonal();
              NewMomentum =
                 NewMomentum.rotate(anglePhotonToNormal-thetaRad,
                                    PerpendicularVectorTheta);
              PerpendicularVectorPhi = 
                                  PerpendicularVectorTheta.cross(NewMomentum);
              NewMomentum = NewMomentum.rotate(-phiRad,PerpendicularVectorPhi);

              // Rotate Polarization too:
              theFacetNormal = (NewMomentum - OldMomentum).unit();
              EdotN = OldPolarization * theFacetNormal;
              NewPolarization = -OldPolarization + (2.*EdotN)*theFacetNormal;
           }
          // Loop checking, 13-Aug-2015, Peter Gumplinger
        } while (NewMomentum * theGlobalNormal <= 0.0);
}

void InstrumentedG4OpBoundaryProcess::DielectricLUTDAVIS()
{
  G4int angindex, random, angleIncident;
  G4double ReflectivityValue, elevation, azimuth, EdotN;
  G4double anglePhotonToNormal;

  G4int LUTbin = OpticalSurface->GetLUTbins();

  G4double rand = G4UniformRand();

  do {

     anglePhotonToNormal = OldMomentum.angle(-theGlobalNormal);
     angleIncident = G4int(std::floor(180/pi*anglePhotonToNormal+0.5));

     ReflectivityValue = OpticalSurface -> GetReflectivityLUTValue(angleIncident);

     if ( rand > ReflectivityValue ) {

        if ( theEfficiency > 0 ) {
           DoAbsorption();
           break;
        }
        else {

           theStatus = Transmission;

           if (angleIncident <= 0.01) {
              NewMomentum = OldMomentum;
              break;

           }

           do {
              random   = G4RandFlat::shootInt(1,LUTbin+1);
              angindex = (((random*2)-1))+angleIncident*LUTbin*2 + 3640000;

              azimuth  = OpticalSurface -> GetAngularDistributionValueLUT(angindex-1);
              elevation= OpticalSurface -> GetAngularDistributionValueLUT(angindex);

           } while ( elevation == 0 && azimuth == 0);

           NewMomentum = -OldMomentum;

           G4ThreeVector v = theGlobalNormal.cross(-NewMomentum);
           G4ThreeVector vNorm = v/v.mag();
           G4ThreeVector u = vNorm.cross(theGlobalNormal);

           u = u *= (std::sin(elevation) * std::cos(azimuth));
           v = vNorm *= (std::sin(elevation) * std::sin(azimuth));
           G4ThreeVector w = theGlobalNormal *= (std::cos(elevation));
           NewMomentum = G4ThreeVector(u+v+w);

           // Rotate Polarization too:
           theFacetNormal = (NewMomentum - OldMomentum).unit();
           EdotN = OldPolarization * theFacetNormal;
           NewPolarization = -OldPolarization + (2.*EdotN)*theFacetNormal;
        }
     }
     else {

        theStatus = LobeReflection;

        if (angleIncident == 0) {
           NewMomentum = -OldMomentum;
           break;
        }

        do {
           random   = G4RandFlat::shootInt(1,LUTbin+1);
           angindex = (((random*2)-1))+(angleIncident-1)*LUTbin*2;

           azimuth   = OpticalSurface -> GetAngularDistributionValueLUT(angindex-1);
           elevation = OpticalSurface -> GetAngularDistributionValueLUT(angindex);
        } while (elevation == 0 && azimuth == 0);

        NewMomentum = -OldMomentum;

        G4ThreeVector v     = theGlobalNormal.cross(-NewMomentum);
        G4ThreeVector vNorm = v/v.mag();
        G4ThreeVector u     = vNorm.cross(theGlobalNormal);

        u = u *= (std::sin(elevation) * std::cos(azimuth));
        v = vNorm *= (std::sin(elevation) * std::sin(azimuth));
        G4ThreeVector w = theGlobalNormal*=(std::cos(elevation));

        NewMomentum = G4ThreeVector(u+v+w);

        // Rotate Polarization too: (needs revision)
        NewPolarization = OldPolarization;
     }
  } while (NewMomentum * theGlobalNormal <= 0.0);
}

void InstrumentedG4OpBoundaryProcess::DielectricDichroic()
{
        // Calculate Angle between Normal and Photon Momentum
        G4double anglePhotonToNormal = OldMomentum.angle(-theGlobalNormal);

        // Round it to closest integer
        G4double angleIncident = std::floor(180/pi*anglePhotonToNormal+0.5);

        if (!DichroicVector) {
           if (OpticalSurface) DichroicVector = OpticalSurface->GetDichroicVector();
        }


        if (DichroicVector) {
           G4double wavelength = h_Planck*c_light/thePhotonMomentum;
           theTransmittance =
             DichroicVector->Value(wavelength/nm,angleIncident,idx,idy)*perCent;
//            G4cout << "wavelength: " << std::floor(wavelength/nm) 
//                                     << "nm" << G4endl;
//            G4cout << "Incident angle: " << angleIncident << "deg" << G4endl;
//            G4cout << "Transmittance: " 
//                   << std::floor(theTransmittance/perCent) << "%" << G4endl;
        } else {
           G4ExceptionDescription ed;
           ed << " InstrumentedG4OpBoundaryProcess/DielectricDichroic(): "
              << " The dichroic surface has no G4Physics2DVector"
              << G4endl;
           G4Exception("InstrumentedG4OpBoundaryProcess::DielectricDichroic", "OpBoun03",
                       FatalException,ed,
                       "A dichroic surface must have an associated G4Physics2DVector");
        }

        if ( !G4BooleanRand(theTransmittance) ) { // Not transmitted, so reflect

           if ( theModel == glisur || theFinish == polished ) {
              DoReflection();
           } else {
              ChooseReflection();
              if ( theStatus == LambertianReflection ) {
                 DoReflection();
              } else if ( theStatus == BackScattering ) {
                 NewMomentum = -OldMomentum;
                 NewPolarization = -OldPolarization;
              } else {
                G4double PdotN, EdotN;
                do {
                   if (theStatus==LobeReflection)
                      theFacetNormal = GetFacetNormal(OldMomentum,theGlobalNormal);
                   PdotN = OldMomentum * theFacetNormal;
                   NewMomentum = OldMomentum - (2.*PdotN)*theFacetNormal;
                  // Loop checking, 13-Aug-2015, Peter Gumplinger
                } while (NewMomentum * theGlobalNormal <= 0.0);
                EdotN = OldPolarization * theFacetNormal;
                NewPolarization = -OldPolarization + (2.*EdotN)*theFacetNormal;
              }
           }

        } else {

           theStatus = Dichroic;
           NewMomentum = OldMomentum;
           NewPolarization = OldPolarization;

        }
}

void InstrumentedG4OpBoundaryProcess::DielectricDielectric()
{

        G4bool Inside = false;
        G4bool Swap = false;

        G4bool SurfaceRoughnessCriterionPass = 1;
        if (theSurfaceRoughness != 0. && Rindex1 > Rindex2) {
           G4double wavelength = h_Planck*c_light/thePhotonMomentum;
           G4double SurfaceRoughnessCriterion =
             std::exp(-std::pow((4*pi*theSurfaceRoughness*Rindex1*cost1/wavelength),2));
           SurfaceRoughnessCriterionPass = 
                                     G4BooleanRand(SurfaceRoughnessCriterion);
        }

        leap:

        G4bool Through = false;
        G4bool Done = false;

        G4double PdotN, EdotN;

        G4ThreeVector A_trans, A_paral, E1pp, E1pl;
        G4double E1_perp, E1_parl;
        G4double s1, s2, E2_perp, E2_parl, E2_total, TransCoeff;
        G4double E2_abs, C_parl, C_perp;
        G4double alpha;

        do {

           if (Through) {
              Swap = !Swap;
              Through = false;
              theGlobalNormal = -theGlobalNormal;
              G4SwapPtr(Material1,Material2);
              G4SwapObj(&Rindex1,&Rindex2);
           }

           if ( theFinish == polished ) {
              theFacetNormal = theGlobalNormal;
           }
           else {
              theFacetNormal =
                             GetFacetNormal(OldMomentum,theGlobalNormal);
           }

           PdotN = OldMomentum * theFacetNormal;
           EdotN = OldPolarization * theFacetNormal;

           cost1 = - PdotN;
           if (std::abs(cost1) < 1.0-kCarTolerance){
              sint1 = std::sqrt(1.-cost1*cost1);
              sint2 = sint1*Rindex1/Rindex2;     // *** Snell's Law ***
           }
           else {
              sint1 = 0.0;
              sint2 = 0.0;
           }

#ifdef DEBUG_PIDX
          if(pidx_dump) std::cout 
              << "DiDi.pidx " << std::setw(4) << pidx  
              << " PIDX " << std::setw(4) << PIDX
              << " OldMomentum (" 
              << " " << std::setw(10) << std::fixed << std::setprecision(5) << OldMomentum.x()
              << " " << std::setw(10) << std::fixed << std::setprecision(5) << OldMomentum.y()
              << " " << std::setw(10) << std::fixed << std::setprecision(5) << OldMomentum.z()
              << ")" 
              << " OldPolarization (" 
              << " " << std::setw(10) << std::fixed << std::setprecision(5) << OldPolarization.x()
              << " " << std::setw(10) << std::fixed << std::setprecision(5) << OldPolarization.y()
              << " " << std::setw(10) << std::fixed << std::setprecision(5) << OldPolarization.z()
              << ")" 
              << " cost1 " << std::setw(10) << std::fixed << std::setprecision(5) << cost1
              << " Rindex1 " << std::setw(10) << std::fixed << std::setprecision(5) << Rindex1
              << " Rindex2 " << std::setw(10) << std::fixed << std::setprecision(5) << Rindex2
              << " sint1 " << std::setw(10) << std::fixed << std::setprecision(5) << sint1 
              << " sint2 " << std::setw(10) << std::fixed << std::setprecision(5) << sint2 
              << std::endl
              ;    
#endif

           if (sint2 >= 1.0) {

              // Simulate total internal reflection

              if (Swap) Swap = !Swap;

              theStatus = TotalInternalReflection;

              if ( !SurfaceRoughnessCriterionPass ) theStatus =
                                                       LambertianReflection;

              if ( theModel == unified && theFinish != polished )
                                                    ChooseReflection();

              if ( theStatus == LambertianReflection ) {
                 DoReflection();
              }
              else if ( theStatus == BackScattering ) {
                 NewMomentum = -OldMomentum;
                 NewPolarization = -OldPolarization;
              }
              else {

                 PdotN = OldMomentum * theFacetNormal;
                 NewMomentum = OldMomentum - (2.*PdotN)*theFacetNormal;
                 EdotN = OldPolarization * theFacetNormal;
                 NewPolarization = -OldPolarization + (2.*EdotN)*theFacetNormal;

              }
           }
           else if (sint2 < 1.0) {

              // Calculate amplitude for transmission (Q = P x N)

              if (cost1 > 0.0) {
                 cost2 =  std::sqrt(1.-sint2*sint2);
              }
              else {
                 cost2 = -std::sqrt(1.-sint2*sint2);
              }

              if (sint1 > 0.0) {
#ifdef DEBUG_PIDX
                 if(pidx_dump) printf("//DiDi pidx %6d : sint1 > 0 \n", pidx );  
#endif

                 A_trans = OldMomentum.cross(theFacetNormal);
                 A_trans = A_trans.unit();
                 E1_perp = OldPolarization * A_trans;
                 E1pp    = E1_perp * A_trans;
                 E1pl    = OldPolarization - E1pp;
                 E1_parl = E1pl.mag();
              }
              else {

#ifdef DEBUG_PIDX
                 if(pidx_dump) printf("//DiDi pidx %6d : NOT:sint1 > 0 : JACKSON NORMAL INCIDENCE  \n", pidx );  
#endif

                 A_trans  = OldPolarization;
                 // Here we Follow Jackson's conventions and we set the
                 // parallel component = 1 in case of a ray perpendicular
                 // to the surface
                 E1_perp  = 0.0;
                 E1_parl  = 1.0;
              }

              s1 = Rindex1*cost1;
              E2_perp = 2.*s1*E1_perp/(Rindex1*cost1+Rindex2*cost2);
              E2_parl = 2.*s1*E1_parl/(Rindex2*cost1+Rindex1*cost2);
              E2_total = E2_perp*E2_perp + E2_parl*E2_parl;
              s2 = Rindex2*cost2*E2_total;

              if (theTransmittance > 0) TransCoeff = theTransmittance;
              else if (cost1 != 0.0) TransCoeff = s2/s1;
              else TransCoeff = 0.0;


#ifdef DEBUG_PIDX
              if(pidx_dump) printf("//DiDi pidx %6d : TransCoeff %10.4f \n", pidx, TransCoeff ); 
#endif

              if ( !G4BooleanRand_TransCoeff(TransCoeff) ) {

                 // Simulate reflection

                 if (Swap) Swap = !Swap;

                 theStatus = FresnelReflection;

                 if ( !SurfaceRoughnessCriterionPass ) theStatus =
                                                          LambertianReflection;

                 if ( theModel == unified && theFinish != polished )
                                                     ChooseReflection();

                 if ( theStatus == LambertianReflection ) {
                    DoReflection();
                 }
                 else if ( theStatus == BackScattering ) {
                    NewMomentum = -OldMomentum;
                    NewPolarization = -OldPolarization;
                 }
                 else {

                    PdotN = OldMomentum * theFacetNormal;
                    NewMomentum = OldMomentum - (2.*PdotN)*theFacetNormal;

                    if (sint1 > 0.0) {   // incident ray oblique

                       E2_parl   = Rindex2*E2_parl/Rindex1 - E1_parl;
                       E2_perp   = E2_perp - E1_perp;
                       E2_total  = E2_perp*E2_perp + E2_parl*E2_parl;
                       A_paral   = NewMomentum.cross(A_trans);
                       A_paral   = A_paral.unit();
                       E2_abs    = std::sqrt(E2_total);
                       C_parl    = E2_parl/E2_abs;
                       C_perp    = E2_perp/E2_abs;

                       NewPolarization = C_parl*A_paral + C_perp*A_trans;

                    }

                    else {               // incident ray perpendicular

                       if (Rindex2 > Rindex1) {
                          NewPolarization = - OldPolarization;
                       }
                       else {
                          NewPolarization =   OldPolarization;
                       }

                    }
                 }
              }
              else { // photon gets transmitted

                // Simulate transmission/refraction

#ifdef DEBUG_PIDX
                if(pidx_dump) printf("//DiDi pidx %6d : TRANSMIT \n", pidx ); 
#endif


                Inside = !Inside;
                Through = true;
                theStatus = FresnelRefraction;

                if (sint1 > 0.0) {      // incident ray oblique

                   alpha = cost1 - cost2*(Rindex2/Rindex1);
                   NewMomentum = OldMomentum + alpha*theFacetNormal;
                   NewMomentum = NewMomentum.unit();
//                   PdotN = -cost2;
                   A_paral = NewMomentum.cross(A_trans);
                   A_paral = A_paral.unit();
                   E2_abs     = std::sqrt(E2_total);
                   C_parl     = E2_parl/E2_abs;
                   C_perp     = E2_perp/E2_abs;

                   NewPolarization = C_parl*A_paral + C_perp*A_trans;

                }
                else {                  // incident ray perpendicular

                   NewMomentum = OldMomentum;
                   NewPolarization = OldPolarization;

                }

#ifdef DEBUG_PIDX
                if(pidx_dump) printf("//DiDi pidx %4d : TRANSMIT NewMom (%10.4f %10.4f %10.4f) NewPol (%10.4f %10.4f %10.4f) \n",
                           pidx, 
                           NewMomentum.x(), NewMomentum.y(), NewMomentum.z(),
                           NewPolarization.x(), NewPolarization.y(), NewPolarization.z()
                      ); 
#endif



              }
           }

           OldMomentum = NewMomentum.unit();
           OldPolarization = NewPolarization.unit();

           if (theStatus == FresnelRefraction) {
              Done = (NewMomentum * theGlobalNormal <= 0.0);
           } 
           else {
              Done = (NewMomentum * theGlobalNormal >= -kCarTolerance);
	   }

        // Loop checking, 13-Aug-2015, Peter Gumplinger
	} while (!Done);

        if (Inside && !Swap) {
          if( theFinish == polishedbackpainted ||
              theFinish == groundbackpainted ) {

              G4double rand = G4UniformRand();
              if ( rand > theReflectivity ) {
                 if (rand > theReflectivity + theTransmittance) {
                    DoAbsorption();
                 } else {
                    theStatus = Transmission;
                    NewMomentum = OldMomentum;
                    NewPolarization = OldPolarization;
                 }
              }
	      else {
                 if (theStatus != FresnelRefraction ) {
                    theGlobalNormal = -theGlobalNormal;
                 }
                 else {
                    Swap = !Swap;
                    G4SwapPtr(Material1,Material2);
                    G4SwapObj(&Rindex1,&Rindex2);
                 }
                 if ( theFinish == groundbackpainted )
                                        theStatus = LambertianReflection;

                 DoReflection();

                 theGlobalNormal = -theGlobalNormal;
                 OldMomentum = NewMomentum;

                 goto leap;
              }
          }
        }
}

// GetMeanFreePath
// ---------------
//
G4double InstrumentedG4OpBoundaryProcess::GetMeanFreePath(const G4Track& ,
                                              G4double ,
                                              G4ForceCondition* condition)
{
  *condition = Forced;

  return DBL_MAX;
}

G4double InstrumentedG4OpBoundaryProcess::GetIncidentAngle() 
{
  G4double PdotN = OldMomentum * theFacetNormal;
  G4double magP= OldMomentum.mag();
  G4double magN= theFacetNormal.mag();
  G4double incidentangle = pi - std::acos(PdotN/(magP*magN));

  return incidentangle;
}

G4double InstrumentedG4OpBoundaryProcess::GetReflectivity(G4double E1_perp,
                                              G4double E1_parl,
                                              G4double incidentangle,
                                              G4double RealRindex,
                                              G4double ImaginaryRindex)
{
  G4complex Reflectivity, Reflectivity_TE, Reflectivity_TM;
  G4complex N1(Rindex1, 0), N2(RealRindex, ImaginaryRindex);
  G4complex CosPhi;

  G4complex u(1,0);           //unit number 1

  G4complex numeratorTE;      // E1_perp=1 E1_parl=0 -> TE polarization
  G4complex numeratorTM;      // E1_parl=1 E1_perp=0 -> TM polarization
  G4complex denominatorTE, denominatorTM;
  G4complex rTM, rTE;

  G4MaterialPropertiesTable* aMaterialPropertiesTable =
                                    Material1->GetMaterialPropertiesTable();
  G4MaterialPropertyVector* aPropertyPointerR =
                      aMaterialPropertiesTable->GetProperty(kREALRINDEX);
  G4MaterialPropertyVector* aPropertyPointerI =
                      aMaterialPropertiesTable->GetProperty(kIMAGINARYRINDEX);
  if (aPropertyPointerR && aPropertyPointerI) {
     G4double RRindex = aPropertyPointerR->Value(thePhotonMomentum);
     G4double IRindex = aPropertyPointerI->Value(thePhotonMomentum);
     N1 = G4complex(RRindex,IRindex);
  }

  // Following two equations, rTM and rTE, are from: "Introduction To Modern
  // Optics" written by Fowles

  CosPhi=std::sqrt(u-((std::sin(incidentangle)*std::sin(incidentangle))*(N1*N1)/(N2*N2)));

  numeratorTE   = N1*std::cos(incidentangle) - N2*CosPhi;
  denominatorTE = N1*std::cos(incidentangle) + N2*CosPhi;
  rTE = numeratorTE/denominatorTE;

  numeratorTM   = N2*std::cos(incidentangle) - N1*CosPhi;
  denominatorTM = N2*std::cos(incidentangle) + N1*CosPhi;
  rTM = numeratorTM/denominatorTM;

  // This is my calculaton for reflectivity on a metalic surface
  // depending on the fraction of TE and TM polarization
  // when TE polarization, E1_parl=0 and E1_perp=1, R=abs(rTE)^2 and
  // when TM polarization, E1_parl=1 and E1_perp=0, R=abs(rTM)^2

  Reflectivity_TE =  (rTE*conj(rTE))*(E1_perp*E1_perp)
                    / (E1_perp*E1_perp + E1_parl*E1_parl);
  Reflectivity_TM =  (rTM*conj(rTM))*(E1_parl*E1_parl)
                    / (E1_perp*E1_perp + E1_parl*E1_parl);
  Reflectivity    = Reflectivity_TE + Reflectivity_TM;

  do {
     if(G4UniformRand()*real(Reflectivity) > real(Reflectivity_TE))
       {iTE = -1;}else{iTE = 1;}
     if(G4UniformRand()*real(Reflectivity) > real(Reflectivity_TM))
       {iTM = -1;}else{iTM = 1;}
    // Loop checking, 13-Aug-2015, Peter Gumplinger
  } while(iTE<0&&iTM<0);

  return real(Reflectivity);

}

void InstrumentedG4OpBoundaryProcess::CalculateReflectivity()
{
  G4double RealRindex =
           PropertyPointer1->Value(thePhotonMomentum);
  G4double ImaginaryRindex =
           PropertyPointer2->Value(thePhotonMomentum);

  // calculate FacetNormal
  if ( theFinish == ground ) {
     theFacetNormal =
               GetFacetNormal(OldMomentum, theGlobalNormal);
  } else {
     theFacetNormal = theGlobalNormal;
  }

  G4double PdotN = OldMomentum * theFacetNormal;
  cost1 = -PdotN;

  if (std::abs(cost1) < 1.0 - kCarTolerance) {
     sint1 = std::sqrt(1. - cost1*cost1);
  } else {
     sint1 = 0.0;
  }

  G4ThreeVector A_trans, A_paral, E1pp, E1pl;
  G4double E1_perp, E1_parl;

  if (sint1 > 0.0 ) {
     A_trans = OldMomentum.cross(theFacetNormal);
     A_trans = A_trans.unit();
     E1_perp = OldPolarization * A_trans;
     E1pp    = E1_perp * A_trans;
     E1pl    = OldPolarization - E1pp;
     E1_parl = E1pl.mag();
  }
  else {
     A_trans  = OldPolarization;
     // Here we Follow Jackson's conventions and we set the
     // parallel component = 1 in case of a ray perpendicular
     // to the surface
     E1_perp  = 0.0;
     E1_parl  = 1.0;
  }

  //calculate incident angle
  G4double incidentangle = GetIncidentAngle();

  //calculate the reflectivity depending on incident angle,
  //polarization and complex refractive

  theReflectivity =
             GetReflectivity(E1_perp, E1_parl, incidentangle,
                                                 RealRindex, ImaginaryRindex);
}

G4bool InstrumentedG4OpBoundaryProcess::InvokeSD(const G4Step* pStep)
{
  G4Step aStep = *pStep;

  aStep.AddTotalEnergyDeposit(thePhotonMomentum);

  G4VSensitiveDetector* sd = aStep.GetPostStepPoint()->GetSensitiveDetector();
  if (sd) return sd->Hit(&aStep);
  else return false;
}


/**
InstrumentedG4OpBoundaryProcess::DoAbsorption
----------------------------------------------


**/

void InstrumentedG4OpBoundaryProcess::DoAbsorption()
{

          LOG(LEVEL) 
               << " PostStepDoIt_count " << PostStepDoIt_count
               << " theEfficiency " << theEfficiency
               ;

              theStatus = Absorption;

              if ( G4BooleanRand_theEfficiency(theEfficiency) ) {

                 // EnergyDeposited =/= 0 means: photon has been detected
                 theStatus = Detection;
                 aParticleChange.ProposeLocalEnergyDeposit(thePhotonMomentum);
              }
              else {
                 aParticleChange.ProposeLocalEnergyDeposit(0.0);
              }

              NewMomentum = OldMomentum;
              NewPolarization = OldPolarization;

//              aParticleChange.ProposeEnergy(0.0);
              aParticleChange.ProposeTrackStatus(fStopAndKill);
}


/**
InstrumentedG4OpBoundaryProcess::DoReflection
-----------------------------------------------

UGLY : depends on and sets theStatus 

**/

void InstrumentedG4OpBoundaryProcess::DoReflection()
{
        if ( theStatus == LambertianReflection ) {

          NewMomentum = U4LambertianRand(theGlobalNormal);
          theFacetNormal = (NewMomentum - OldMomentum).unit();

        }
        else if ( theFinish == ground ) {

          theStatus = LobeReflection;
          if ( PropertyPointer1 && PropertyPointer2 ){
          } else {
             theFacetNormal =
                 GetFacetNormal(OldMomentum,theGlobalNormal);
          }
          G4double PdotN = OldMomentum * theFacetNormal;
          NewMomentum = OldMomentum - (2.*PdotN)*theFacetNormal;

        }
        else {

          theStatus = SpikeReflection;
          theFacetNormal = theGlobalNormal;
          G4double PdotN = OldMomentum * theFacetNormal;
          NewMomentum = OldMomentum - (2.*PdotN)*theFacetNormal;

        }
        G4double EdotN = OldPolarization * theFacetNormal;
        NewPolarization = -OldPolarization + (2.*EdotN)*theFacetNormal;
}



#ifdef DEBUG_PIDX
const int  InstrumentedG4OpBoundaryProcess::PIDX  = std::atoi( getenv("PIDX") ? getenv("PIDX") : "-1" );  
#endif




G4bool InstrumentedG4OpBoundaryProcess::G4BooleanRand(const G4double prob) const
{
  G4double u = G4UniformRand() ; 
#ifdef DEBUG_PIDX
  if(pidx_dump) printf("//InstrumentedG4OpBoundaryProcess::G4BooleanRand pidx %6d prob %10.5f u %10.5f u < prob %d \n", pidx, prob,u, (u < prob) );  
#endif
  return u < prob  ; 
}

G4bool InstrumentedG4OpBoundaryProcess::G4BooleanRand_TransCoeff(const G4double prob) const
{
  G4double u = G4UniformRand() ; 

#ifdef DEBUG_TAG
   SEvt::AddTag(U4Stack_BoundaryDiDiTransCoeff, u ); 
#endif   

#ifdef DEBUG_PIDX
  if(pidx_dump) printf("//InstrumentedG4OpBoundaryProcess::G4BooleanRand_TransCoff pidx %6d prob %10.5f u %10.5f u < prob %d \n", pidx, prob,u, (u < prob) );  
#endif
  return u < prob  ; 
}

G4bool InstrumentedG4OpBoundaryProcess::G4BooleanRand_theEfficiency(const G4double prob) const
{
  G4double u = G4UniformRand() ; 
#ifdef DEBUG_TAG
   SEvt::AddTag(U4Stack_AbsorptionEffDetect, u ); 
#endif   
#ifdef DEBUG_PIDX
  if(pidx_dump) printf("//InstrumentedG4OpBoundaryProcess::G4BooleanRand_theEfficiency pidx %6d prob %10.5f u %10.5f u < prob %d \n", pidx, prob,u, (u < prob) );  
#endif
  return u < prob  ; 
}


G4bool InstrumentedG4OpBoundaryProcess::G4BooleanRand_theReflectivity(const G4double prob) const
{
  G4double u = G4UniformRand() ; 
#ifdef DEBUG_PIDX
  if(pidx_dump) printf("//InstrumentedG4OpBoundaryProcess::G4BooleanRand_theReflectivity pidx %6d prob %10.5f u %10.5f u < prob %d \n", pidx, prob,u, (u < prob) );  
#endif
  return u < prob  ; 
}



G4bool InstrumentedG4OpBoundaryProcess::IsApplicable(const G4ParticleDefinition& 
                                                       aParticleType)
{
   return ( &aParticleType == G4OpticalPhoton::OpticalPhoton() );
}

G4OpBoundaryProcessStatus InstrumentedG4OpBoundaryProcess::GetStatus() const
{
   return theStatus;
}

void InstrumentedG4OpBoundaryProcess::SetInvokeSD(G4bool flag)
{
  fInvokeSD = flag;
}

void InstrumentedG4OpBoundaryProcess::ChooseReflection()
{
                 G4double rand = G4UniformRand();
#ifdef DEBUG_TAG
                 SEvt::AddTag( U4Stack_ChooseReflection, rand ); 
#endif

                 if ( rand >= 0.0 && rand < prob_ss ) {
                    theStatus = SpikeReflection;
                    theFacetNormal = theGlobalNormal;
                 }
                 else if ( rand >= prob_ss &&
                           rand <= prob_ss+prob_sl) {
                    theStatus = LobeReflection;
                 }
                 else if ( rand > prob_ss+prob_sl &&
                           rand < prob_ss+prob_sl+prob_bs ) {
                    theStatus = BackScattering;
                 }
                 else {
                    theStatus = LambertianReflection;
                 }
}


