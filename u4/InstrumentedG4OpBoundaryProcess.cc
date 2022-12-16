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


#ifdef WITH_PMTFASTSIM
#include "CustomBoundary.h"
#include "CustomART.h"
#endif

#include "SLOG.hh"
#include "spho.h"
#include "STrackInfo.h"
#include "U4OpticalSurface.h"
#include "U4OpBoundaryProcessStatus.h"
#include "U4MaterialPropertiesTable.h"



#include "U4UniformRand.h"
NP* U4UniformRand::UU = nullptr ;  
// UU gets set by U4Recorder::saveOrLoadStates when doing single photon reruns

#ifdef DEBUG_PIDX

    #include "scuda.h"
    #include "squad.h"
    #include "SEvt.hh"
    #include "SSys.hh"
    #include "U4PhotonInfo.h"  // TODO: should be STrackInfo<spho> now
    const int InstrumentedG4OpBoundaryProcess::PIDX = SSys::getenvint("PIDX", -1) ; 

#endif


#ifdef DEBUG_TAG
#include "U4Stack.h"
#include "SEvt.hh"

const plog::Severity InstrumentedG4OpBoundaryProcess::LEVEL = SLOG::EnvLevel("InstrumentedG4OpBoundaryProcess", "DEBUG" ); 

const bool InstrumentedG4OpBoundaryProcess::FLOAT  = getenv("InstrumentedG4OpBoundaryProcess_FLOAT") != nullptr ;

//void InstrumentedG4OpBoundaryProcess::ResetNumberOfInteractionLengthLeft(){ G4VProcess::ResetNumberOfInteractionLengthLeft(); }
void InstrumentedG4OpBoundaryProcess::ResetNumberOfInteractionLengthLeft()
{
    G4double u0 = G4UniformRand() ; 

    bool burn_enabled = true ; 
    if(burn_enabled)
    {
        int u0_idx = U4UniformRand::Find(u0, SEvt::UU ) ;  
        int u0_idx_delta = u0_idx - m_u0_idx ; 
 
        LOG(LEVEL) 
            << U4UniformRand::Desc(u0, SEvt::UU ) 
            << " u0_idx " << u0_idx 
            << " u0_idx_delta " << u0_idx_delta 
            << " BOP.RESET " 
            ; 

        int uu_burn = SEvt::UU_BURN ? SEvt::UU_BURN->ifind2D<int>(u0_idx, 0, 1 ) : -1  ; 

        if( uu_burn > 0 )
        {
            u0 = U4UniformRand::Burn(uu_burn);
            u0_idx = U4UniformRand::Find(u0, SEvt::UU ) ;  
            u0_idx_delta = u0_idx - m_u0_idx ;   

            LOG(LEVEL) 
                << U4UniformRand::Desc(u0, SEvt::UU ) 
                << " u0_idx " << u0_idx 
                << " u0_idx_delta " << u0_idx_delta 
                << " after uu_burn " << uu_burn
                << " BOP.RESET " 
                ; 
        } 

        m_u0 = u0 ; 
        m_u0_idx = u0_idx ; 
        m_u0_idx_delta = u0_idx_delta ; 
    }

    SEvt::AddTag( U4Stack_BoundaryDiscreteReset, u0 );  

    if(FLOAT)
    {   
        float f = -1.f*std::log( float(u0) ) ;   
        theNumberOfInteractionLengthLeft = f ; 
    }   
    else
    {   
        theNumberOfInteractionLengthLeft = -1.*G4Log(u0) ;   
    }   
    theInitialNumberOfInteractionLength = theNumberOfInteractionLengthLeft; 

}
#endif

#ifdef WITH_PMTFASTSIM
double InstrumentedG4OpBoundaryProcess::getU0() const
{
    return m_u0 ; 
}
int InstrumentedG4OpBoundaryProcess::getU0_idx() const
{
    return m_u0_idx ; 
}
const double* InstrumentedG4OpBoundaryProcess::getRecoveredNormal() const 
{
    return (const double*)&theRecoveredNormal ;
}

/**
InstrumentedG4OpBoundaryProcess::getCustomStatus
--------------------------------------------------

HMM: actually makes more sense to hold a more 
general status within InstrumentedG4OpBoundaryProcess 
which describes whether the custom_boundary was used or not 
and which is informative on why custom boundary used/not-used.
That is more relevant than the R/T/A which is already described
by the standard theStatus. 

**/

char InstrumentedG4OpBoundaryProcess::getCustomStatus() const 
{
    return theCustomStatus ; 
    //return m_custom_boundary ? m_custom_boundary->customStatus : '-' ; 
}
void InstrumentedG4OpBoundaryProcess::Save(const char* fold) // static
{
    CustomBoundary<JPMT>::Save(fold); 
}

#endif


InstrumentedG4OpBoundaryProcess::InstrumentedG4OpBoundaryProcess(const G4String& processName, G4ProcessType type) 
    : 
    G4VDiscreteProcess(processName, type)
#ifdef WITH_PMTFASTSIM
    ,SOpBoundaryProcess(processName.c_str())
#endif
    ,theCustomStatus('U')
#ifdef WITH_PMTFASTSIM
    ,m_custom_boundary(new CustomBoundary<JPMT>(
                  NewMomentum,
                  NewPolarization,
                  aParticleChange,
                  theStatus,
                  theGlobalPoint,
                  OldMomentum,
                  OldPolarization,
                  theRecoveredNormal,
                  thePhotonMomentum))

    ,m_custom_art(new CustomART<JPMT>(
                  theTransmittance,
                  theReflectivity,
                  theEfficiency,
                  theGlobalPoint,
                  OldMomentum,
                  OldPolarization,
                  theRecoveredNormal,
                  thePhotonMomentum))
    ,m_u0(-1.)
    ,m_u0_idx(-1)
#endif
    ,PostStepDoIt_count(-1)
{
    LOG(LEVEL) << " processName " << GetProcessName()  ; 

    SetProcessSubType(fOpBoundary);

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

    theRecoveredNormal.set(0.,0.,0.); 

    OpticalSurface = NULL;

    kCarTolerance = G4GeometryTolerance::GetInstance()->GetSurfaceTolerance(); 

    iTE = iTM = 0;
    thePhotonMomentum = 0.;
    Rindex1 = Rindex2 = 1.;
    cost1 = cost2 = sint1 = sint2 = 0.;

    idx = idy = 0;
    DichroicVector = NULL;

    fInvokeSD = true;
}

InstrumentedG4OpBoundaryProcess::~InstrumentedG4OpBoundaryProcess(){}


/**
InstrumentedG4OpBoundaryProcess::PostStepDoIt
-----------------------------------------------

Wrapper to help with the spagetti mush  

**/

G4VParticleChange* InstrumentedG4OpBoundaryProcess::PostStepDoIt(const G4Track& aTrack, const G4Step& aStep)
{
#ifdef WITH_PMTFASTSIM
    PostStepDoIt_count += 1 ;  
    spho* label = STrackInfo<spho>::GetRef(&aTrack) ; 

    LOG(LEVEL) 
        << "[ "
        << " PostStepDoIt_count " << PostStepDoIt_count
        << " label.desc " << label->desc()
        ; 
#endif
    G4VParticleChange* change = PostStepDoIt_(aTrack, aStep) ; 

#ifdef WITH_PMTFASTSIM
    LOG(LEVEL) << "] " << PostStepDoIt_count  ; 
#endif
    return change ; 
}

/**
InstrumentedG4OpBoundaryProcess::PostStepDoIt_
-------------------------------------------------

UNBELIEVABLY ABYSMAL CODING EVEN FOR GEANT4 


head
    Material1, Material2, thePhotonMomentum, OldMomentum, OldPolarization
theGlobalNormal
    note the flip 
Rindex1  
    from Material1
Defaults
    theReflectivity=1, theEfficiency=0, theTransmittance=0
Surface
    border or skin  
opsu
    detect if CustomART enabled based on OpticalSurface name
OpticalSurface
    type, theModel, theFinish

    OpticalSurface.mpt

        OpticalSurface.mpt.backpainted
            Rindex2 from OpticalSurface.mpt   
            OpticalSurface with finish:polishedbackpainted/groundbackpainted
            require mpt with RINDEX to avoid NoRINDEX fStopAndKill

        OpticalSurface.mpt.theReflectivity,theTransmittance,theEfficiency
            sets theReflectivity
            sets theTransmittance
            sets theEfficiency

        OpticalSurface.mpt.CustomART
            custom:true call CustomART  
         
        OpticalSurface.mpt.unified
            theModel:unified sets prob_sl/prob_ss/prob_bs 

        OpticalSurface.no-mpt.backpainted
            fStopAndKill with ProposeLocalEnergyDeposit(thePhotonMomentum)  

didi.polished/ground
    get Material2.mpt (not OpticalSurface.mpt)
    get Material2.mpt.Rindex2 otherwise NoRINDEX fStopAndKill

type_switch
    type_switch.dime
        DielectricMetal
    type_switch.didi
        type_switch.didi.backpainted
             DielectricDielectric
        type_switch.didi.not_backpainted 

             rand-3-way depending on 

             theReflectivity 
             theReflectivity+theTransmittance         

                    R            R+T
             +------|-------------|----------------+
              R:refl    T:trans         A:abs               

             type_switch.didi.not_backpainted.A
                 DoAbsorption 

             type_switch.didi.not_backpainted.T
                 set theStatus=Transmission 
                 [UNNATURAL UNCHANGED DIR, POL]

                 NB: theTransmittance default is zero, so this needs a 
                 TRANSMITTANCE property to get it to happen

             type_switch.didi.not_backpainted.R
                 theFinish:(polished/ground)frontpainted
                     ground: set theStatus=LambertianReflection 
                     DoReflection
                 not-frontpainted
                     DielectricDielectric  

tail
    aParticleChange NewMomentum, NewPolarization after .unit()


**/

G4VParticleChange* InstrumentedG4OpBoundaryProcess::PostStepDoIt_(const G4Track& aTrack, const G4Step& aStep)
{
    //[head
#ifdef DEBUG_PIDX
    // U4PhotonInfo::GetIndex is picking up the index from the label set 
    // in U4Recorder::PreUserTrackingAction_Optical for initially unlabelled input photons
    pidx = U4PhotonInfo::GetIndex(&aTrack);    // TODO: now needs to be STrackInfo<spho>
    pidx_dump = pidx == PIDX ; 
#endif
    theStatus = Undefined;
    theCustomStatus = 'B' ; 

    aParticleChange.Initialize(aTrack);
    aParticleChange.ProposeVelocity(aTrack.GetVelocity());

    // Get hyperStep from  G4ParallelWorldProcess
    //  NOTE: PostSetpDoIt of this process should be
    //        invoked after G4ParallelWorldProcess!

    const G4Step* pStep = &aStep;
    const G4Step* hStep = G4ParallelWorldProcess::GetHyperStep();
        
    if (hStep) pStep = hStep;
    G4bool isOnBoundary = (pStep->GetPostStepPoint()->GetStepStatus() == fGeomBoundary);

    if (isOnBoundary) 
    {
        Material1 = pStep->GetPreStepPoint()->GetMaterial();
        Material2 = pStep->GetPostStepPoint()->GetMaterial();
    } 
    else 
    {
        theStatus = NotAtBoundary;
        if ( verboseLevel > 0) BoundaryProcessVerbose();
        return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
    }

    G4VPhysicalVolume* thePrePV  = pStep->GetPreStepPoint() ->GetPhysicalVolume(); 
    G4VPhysicalVolume* thePostPV = pStep->GetPostStepPoint()->GetPhysicalVolume(); 
    G4bool haveEnteredDaughter= (thePostPV->GetMotherLogical() == thePrePV ->GetLogicalVolume()); // SCB

    LOG(LEVEL)
        << " PostStepDoIt_count " << PostStepDoIt_count
        << " thePrePV " << ( thePrePV ? thePrePV->GetName() : "-" )
        << " thePostPV " << ( thePostPV ? thePostPV->GetName() : "-" )
        << " haveEnteredDaughter " << haveEnteredDaughter
        << " kCarTolerance/2 " << kCarTolerance/2
        ;

    if (aTrack.GetStepLength()<=kCarTolerance/2)
    {
        theStatus = StepTooSmall;
        return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
    }

    const G4DynamicParticle* aParticle = aTrack.GetDynamicParticle();
    thePhotonMomentum = aParticle->GetTotalMomentum();
    OldMomentum       = aParticle->GetMomentumDirection();
    OldPolarization   = aParticle->GetPolarization();

    //]head

    //[theGlobalNorml
    theGlobalPoint = pStep->GetPostStepPoint()->GetPosition();

    G4bool valid;
    // Use the new method for Exit Normal in global coordinates,
    // which provides the normal more reliably.
    // ID of Navigator which limits step

    G4int hNavId = G4ParallelWorldProcess::GetHypNavigatorID();
    std::vector<G4Navigator*>::iterator iNav =
                G4TransportationManager::GetTransportationManager()->
                                         GetActiveNavigatorsIterator();
    theGlobalExitNormal =
                   (iNav[hNavId])->GetGlobalExitNormal(theGlobalPoint,&valid);

    // theGlobalExitNormal is already oriented by G4Navigator to point from vol1 -> vol2 
    // so try to undo that flip by G4Navigator in order to recover the original geometry 
    // normal that is independent of G4Track direction

    theRecoveredNormal = ( haveEnteredDaughter ? -1. : 1. )* theGlobalExitNormal  ; 



    theGlobalNormal = theGlobalExitNormal ;  


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

    if (valid) 
    {
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



    if (OldMomentum * theGlobalNormal > 0.0) 
    {
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
    //]theGlobalNorml


    //[Rindex1
    G4MaterialPropertiesTable* aMaterialPropertiesTable;
    G4MaterialPropertyVector* Rindex;

    aMaterialPropertiesTable = Material1->GetMaterialPropertiesTable();
    if (aMaterialPropertiesTable) 
    {
        Rindex = aMaterialPropertiesTable->GetProperty(kRINDEX);
    }
    else 
    {
        theStatus = NoRINDEX;
        if ( verboseLevel > 0) BoundaryProcessVerbose();
        aParticleChange.ProposeLocalEnergyDeposit(thePhotonMomentum);
        aParticleChange.ProposeTrackStatus(fStopAndKill);
        return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
    }

    if (Rindex) 
    {
        Rindex1 = Rindex->Value(thePhotonMomentum);
    }
    else 
    {
        theStatus = NoRINDEX;
        if ( verboseLevel > 0) BoundaryProcessVerbose();
        aParticleChange.ProposeLocalEnergyDeposit(thePhotonMomentum);
        aParticleChange.ProposeTrackStatus(fStopAndKill);
        return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
    }
    //]Rindex1
    // SCB: UNBELIEVABLY POOR CODING STYLE : POINTLESS REPETITION OF BLOCKS OF CODE

    //[Defaults
    theReflectivity =  1.;
    theEfficiency   =  0.;
    theTransmittance = 0.;

    theSurfaceRoughness = 0.;

    theModel = glisur;
    theFinish = polished;

    G4SurfaceType type = dielectric_dielectric;

    //]Defaults

    //[Surface
    Rindex = NULL;
    OpticalSurface = NULL;

    G4LogicalSurface* Surface = NULL;
    Surface = G4LogicalBorderSurface::GetSurface(thePrePV, thePostPV);
    if (Surface == NULL)
    {
        G4bool enteredDaughter= (thePostPV->GetMotherLogical() == thePrePV ->GetLogicalVolume());
        if(enteredDaughter)
        {
            Surface = G4LogicalSkinSurface::GetSurface(thePostPV->GetLogicalVolume());
            if(Surface == NULL) Surface = G4LogicalSkinSurface::GetSurface(thePrePV->GetLogicalVolume());
        }
        else 
        {
            Surface = G4LogicalSkinSurface::GetSurface(thePrePV->GetLogicalVolume());
            if(Surface == NULL) Surface = G4LogicalSkinSurface::GetSurface(thePostPV->GetLogicalVolume());
        }
    }
    //]Surface


    if (Surface) OpticalSurface = dynamic_cast <G4OpticalSurface*> (Surface->GetSurfaceProperty()); 
    LOG(LEVEL) 
        << " PostStepDoIt_count " << PostStepDoIt_count
        << " Surface " << ( Surface ? Surface->GetName() : "-" ) 
        << " OpticalSurface " << ( OpticalSurface ? OpticalSurface->GetName() : "-" ) 
        ; 

    //[OpticalSurface
    if (OpticalSurface) 
    {
        const char* OpticalSurfaceName = OpticalSurface->GetName().c_str() ;  // GetName by ref, so not transient 
        LOG(LEVEL) 
            << " PostStepDoIt_count " << PostStepDoIt_count << " "
            << " OpticalSurfaceName " << OpticalSurfaceName
            << U4OpticalSurface::Desc(OpticalSurface) 
            ; 

        type      = OpticalSurface->GetType();
        theModel  = OpticalSurface->GetModel();
        theFinish = OpticalSurface->GetFinish();

        aMaterialPropertiesTable = OpticalSurface->GetMaterialPropertiesTable(); 

        //[OpticalSurface.mpt
        if (aMaterialPropertiesTable) 
        {
            /*
            LOG(LEVEL) 
                << " PostStepDoIt_count " << PostStepDoIt_count << " " 
                << U4MaterialPropertiesTable::Desc(aMaterialPropertiesTable) 
                ; 
            */

            //[OpticalSurface.mpt.backpainted
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
            //]OpticalSurface.mpt.backpainted

            //[OpticalSurface.mpt.theReflectivity,theTransmittance,theEfficiency
            PropertyPointer  = aMaterialPropertiesTable->GetProperty(kREFLECTIVITY); 
            PropertyPointer1 = aMaterialPropertiesTable->GetProperty(kREALRINDEX); 
            PropertyPointer2 = aMaterialPropertiesTable->GetProperty(kIMAGINARYRINDEX); 

            /*
            LOG(LEVEL) 
                << " PostStepDoIt_count " << PostStepDoIt_count
                << " PropertyPointer.kREFLECTIVITY " << PropertyPointer
                << " PropertyPointer1.kREALRINDEX " << PropertyPointer1 
                << " PropertyPointer2.kIMAGINARYRINDEX " << PropertyPointer2
                ;
            */

            iTE = 1;
            iTM = 1;

            if (PropertyPointer) 
            {
                theReflectivity = PropertyPointer->Value(thePhotonMomentum); 
            }
            else if (PropertyPointer1 && PropertyPointer2) 
            {
                CalculateReflectivity();  // sets theReflectivity 
            }


            PropertyPointer = aMaterialPropertiesTable->GetProperty(kEFFICIENCY); 
            if (PropertyPointer) 
            {
                theEfficiency = PropertyPointer->Value(thePhotonMomentum); 
            }

            PropertyPointer = aMaterialPropertiesTable->GetProperty(kTRANSMITTANCE); 
            if (PropertyPointer) 
            {
                theTransmittance = PropertyPointer->Value(thePhotonMomentum); 
            }
            //]OpticalSurface.mpt.theReflectivity,theTransmittance,theEfficiency

            //[OpticalSurface.mpt.CustomBoundary
#ifdef WITH_PMTFASTSIM
            theCustomStatus = m_custom_boundary->maybe_doIt( OpticalSurfaceName, aTrack, aStep );  
#else
            theCustomStatus = 'X' ; 
#endif
            //]OpticalSurface.mpt.CustomBoundary

            LOG(LEVEL)
                << " PostStepDoIt_count " << PostStepDoIt_count
                << " theReflectivity " << theReflectivity
                << " theEfficiency " << theEfficiency
                << " theTransmittance " << theTransmittance
                << " theCustomStatus " << theCustomStatus
                ; 

            if (aMaterialPropertiesTable->ConstPropertyExists("SURFACEROUGHNESS"))
                theSurfaceRoughness = aMaterialPropertiesTable-> GetConstProperty(kSURFACEROUGHNESS); 

            //[OpticalSurface.mpt.unified
            if ( theModel == unified ) 
            {
                PropertyPointer = aMaterialPropertiesTable->GetProperty(kSPECULARLOBECONSTANT); 
                prob_sl = PropertyPointer ? PropertyPointer->Value(thePhotonMomentum) : 0.0 ; 

                PropertyPointer = aMaterialPropertiesTable->GetProperty(kSPECULARSPIKECONSTANT); 
                prob_ss = PropertyPointer ? PropertyPointer->Value(thePhotonMomentum) : 0.0 ; 

                PropertyPointer = aMaterialPropertiesTable->GetProperty(kBACKSCATTERCONSTANT); 
                prob_bs = PropertyPointer ? PropertyPointer->Value(thePhotonMomentum) : 0.0 ; 
            }
            //]OpticalSurface.mpt.unified
        }
        //]OpticalSurface.mpt
        //[OpticalSurface.no-mpt.backpainted
        else if (theFinish == polishedbackpainted || theFinish == groundbackpainted ) 
        {
            aParticleChange.ProposeLocalEnergyDeposit(thePhotonMomentum);
            aParticleChange.ProposeTrackStatus(fStopAndKill);
            return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
        }
        //]OpticalSurface.no-mpt.backpainted
    }
    //]OpticalSurface

    LOG(LEVEL) 
        << " PostStepDoIt_count " << PostStepDoIt_count
        << " after OpticalSurface if "  
        ;


    //[didi.polished/ground
    if (type == dielectric_dielectric ) 
    {
        if (theFinish == polished || theFinish == ground ) 
        {
            if (Material1 == Material2)
            {
                theStatus = SameMaterial;
                if ( verboseLevel > 0) BoundaryProcessVerbose();
                return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
            }
            aMaterialPropertiesTable = Material2->GetMaterialPropertiesTable(); 

            if (aMaterialPropertiesTable) Rindex = aMaterialPropertiesTable->GetProperty(kRINDEX); 
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
        LOG(LEVEL) 
            << " PostStepDoIt_count " << PostStepDoIt_count
            << " didi.polished/ground " 
            << " Rindex2 " << Rindex2 
            ;  
    }
    //]didi.polished/ground


    //[type_switch 
#ifdef WITH_PMTFASTSIM
    if(strchr("ARTD", theCustomStatus)!=nullptr) 
    {
        LOG(LEVEL) << "CustomBoundary_doneIt : SKIP TYPE_SWITCH " ;  
    }
    else 
#endif
    if (type == dielectric_metal) 
    {
        //[type_switch.dime
        DielectricMetal();
        //]type_switch.dime
    }
    else if (type == dielectric_LUT) 
    {
        DielectricLUT();
    }
    else if (type == dielectric_LUTDAVIS) 
    {
        DielectricLUTDAVIS();
    }
    else if (type == dielectric_dichroic) 
    {
        DielectricDichroic();
    }
    else if (type == dielectric_dielectric) 
    {
        //[type_switch.didi
        if ( theFinish == polishedbackpainted || theFinish == groundbackpainted ) 
        {
            //[type_switch.didi.backpainted
            DielectricDielectric();
            //]type_switch.didi.backpainted
        }
        else
        {   
            //[type_switch.didi.not_backpainted
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
            LOG_IF(LEVEL, pidx_dump) 
                << " DiDi0 " 
                << " pidx " << std::setw(6) << pidx 
                << " rand " << std::setw(10) << std::fixed << std::setprecision(5) << rand  
                << " theReflectivity " << std::setw(10) << std::fixed << std::setprecision(4) << theReflectivity
                << " rand > theReflectivity  " << (rand > theReflectivity )
                ;
#endif
            if ( rand > theReflectivity )
            {
                if (rand > theReflectivity + theTransmittance) 
                {
                    //[type_switch.didi.not_backpainted.A
                    DoAbsorption();
                    //]type_switch.didi.not_backpainted.A
                }
                else 
                {
                    //[type_switch.didi.not_backpainted.T
                    theStatus = Transmission;
                    NewMomentum = OldMomentum;
                    NewPolarization = OldPolarization;
                    LOG(LEVEL) << " mystifying Transmission with Mom and Pol unchanged ? " ; 
                    //]type_switch.didi.not_backpainted.T
                }
            }
            else 
            {
                //[type_switch.didi.not_backpainted.R
                if ( theFinish == polishedfrontpainted ) 
                {
                    DoReflection();
                }
                else if ( theFinish == groundfrontpainted ) 
                {
                    theStatus = LambertianReflection;
                    DoReflection();
                }
                else 
                {
                    DielectricDielectric();
                }
                //]type_switch.didi.not_backpainted.R
            }
            //]type_switch.didi.not_backpainted
        }   
        //[type_switch.didi
    }
    else 
    {
        //[type_switch.illegal
        G4cerr << " Error: G4BoundaryProcess: illegal boundary type " << G4endl;
        return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
        //]type_switch.illegal
    }
    //]type switch 

    //[tail 
    NewMomentum = NewMomentum.unit();
    NewPolarization = NewPolarization.unit();

    aParticleChange.ProposeMomentumDirection(NewMomentum);
    aParticleChange.ProposePolarization(NewPolarization);

    if ( theStatus == FresnelRefraction || theStatus == Transmission ) 
    {
        G4MaterialPropertyVector* groupvel = Material2->GetMaterialPropertiesTable()->GetProperty(kGROUPVEL);
        G4double finalVelocity = groupvel->Value(thePhotonMomentum);
        aParticleChange.ProposeVelocity(finalVelocity);
    }
    if ( theStatus == Detection && fInvokeSD ) InvokeSD(pStep);

    // ]tail 
    return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
}


/**
InstrumentedG4OpBoundaryProcess::GetFacetNormal
------------------------------------------------


This function code alpha to a random value taken from the
distribution p(alpha) = g(alpha; 0, sigma_alpha)*std::sin(alpha),
for alpha > 0 and alpha < 90, where g(alpha; 0, sigma_alpha)
is a gaussian distribution with mean 0 and standard deviation
sigma_alpha.  


**/

G4ThreeVector InstrumentedG4OpBoundaryProcess::GetFacetNormal(
     const G4ThreeVector& Momentum, const G4ThreeVector&  Normal ) const 
{
    G4ThreeVector FacetNormal;

    if (theModel == unified || theModel == LUT || theModel== DAVIS) 
    {
        G4double alpha;
        G4double sigma_alpha = 0.0;
        if (OpticalSurface) sigma_alpha = OpticalSurface->GetSigmaAlpha();

        if (sigma_alpha == 0.0) return FacetNormal = Normal;

        G4double f_max = std::min(1.0,4.*sigma_alpha);

        G4double phi, SinAlpha, CosAlpha, SinPhi, CosPhi, unit_x, unit_y, unit_z;
        G4ThreeVector tmpNormal;

        do 
        {
            do 
            {
                 alpha = G4RandGauss::shoot(0.0,sigma_alpha);
                 // Loop checking, 13-Aug-2015, Peter Gumplinger
            } 
            while (G4UniformRand()*f_max > std::sin(alpha) || alpha >= halfpi );

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
        } 
        while (Momentum * FacetNormal >= 0.0);
    }
    else 
    {
        G4double  polish = OpticalSurface ? OpticalSurface->GetPolish() : 1.0 ;
        if (polish < 1.0) 
        {
            do 
            {
                G4ThreeVector smear;
                do 
                {
                    smear.setX(2.*G4UniformRand()-1.0);
                    smear.setY(2.*G4UniformRand()-1.0);
                    smear.setZ(2.*G4UniformRand()-1.0);
                    // Loop checking, 13-Aug-2015, Peter Gumplinger
                } 
                while (smear.mag()>1.0);
                smear = (1.-polish) * smear;
                FacetNormal = Normal + smear;
                // Loop checking, 13-Aug-2015, Peter Gumplinger
            } 
            while (Momentum * FacetNormal >= 0.0);
            FacetNormal = FacetNormal.unit();
        }
        else 
        {
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

    do 
    {
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


        if ( rand > theReflectivity && n == 1 ) 
        {
            if (rand > theReflectivity + theTransmittance) 
            {
                DoAbsorption();
            } 
            else 
            {
                theStatus = Transmission;
                NewMomentum = OldMomentum;
                NewPolarization = OldPolarization;
            }
            LOG(LEVEL) << " rand > theReflectivity && n == 1  break " ; 
            break;
        }
        else 
        {
            if (PropertyPointer1 && PropertyPointer2) 
            {
                if ( n > 1 ) 
                {
                    CalculateReflectivity();
                    if ( !G4BooleanRand_theReflectivity(theReflectivity) ) 
                    {
                        DoAbsorption();
                        break;
                    }
                }
            }

            if ( theModel == glisur || theFinish == polished ) 
            {
                DoReflection();
            } 
            else 
            {
                if ( n == 1 ) ChooseReflection();
                                                                                
                if ( theStatus == LambertianReflection ) 
                {
                    DoReflection();
                }
                else if ( theStatus == BackScattering ) 
                {
                    NewMomentum = -OldMomentum;
                    NewPolarization = -OldPolarization;
                }
                else 
                {
                    if(theStatus==LobeReflection)
                    {
                        if ( PropertyPointer1 && PropertyPointer2 )
                        {
                        } 
                        else 
                        {
                            theFacetNormal = GetFacetNormal(OldMomentum,theGlobalNormal);
                        }
                    }

                    PdotN = OldMomentum * theFacetNormal;
                    NewMomentum = OldMomentum - (2.*PdotN)*theFacetNormal;
                    EdotN = OldPolarization * theFacetNormal;

                    if (sint1 > 0.0 ) 
                    {
                        A_trans = OldMomentum.cross(theFacetNormal);
                        A_trans = A_trans.unit();
                    } 
                    else 
                    {
                        A_trans  = OldPolarization;
                    }
                    A_paral   = NewMomentum.cross(A_trans);
                    A_paral   = A_paral.unit();

                    if(iTE>0&&iTM>0) 
                    {
                        NewPolarization = -OldPolarization + (2.*EdotN)*theFacetNormal;
                    } 
                    else if (iTE>0) 
                    {
                        NewPolarization = -A_trans;
                    } 
                    else if (iTM>0) 
                    {
                        NewPolarization = -A_paral;
                    }
                }
            }

            //LOG(LEVEL) << " dime: Old<-New ? UGLY " ; 
            OldMomentum = NewMomentum;
            OldPolarization = NewPolarization;
	    }
        //LOG(LEVEL) << " while check " ; 
          // Loop checking, 13-Aug-2015, Peter Gumplinger
	} while (NewMomentum * theGlobalNormal < 0.0);
    //LOG(LEVEL) << " dime: after while  " ; 
}

void InstrumentedG4OpBoundaryProcess::DielectricLUT()
{
    G4int thetaIndex, phiIndex;
    G4double AngularDistributionValue, thetaRad, phiRad, EdotN;
    G4ThreeVector PerpendicularVectorTheta, PerpendicularVectorPhi;

    theStatus = G4OpBoundaryProcessStatus(G4int(theFinish) + (G4int(NoRINDEX)-G4int(groundbackpainted))); 

    G4int thetaIndexMax = OpticalSurface->GetThetaIndexMax();
    G4int phiIndexMax   = OpticalSurface->GetPhiIndexMax();

    G4double rand;

    do 
    {
        rand = G4UniformRand();
        if ( rand > theReflectivity ) 
        {
            if (rand > theReflectivity + theTransmittance) 
            {
                DoAbsorption();
            } 
            else 
            {
                theStatus = Transmission;
                NewMomentum = OldMomentum;
                NewPolarization = OldPolarization;
            }
            break;
        }
        else 
        {
            // Calculate Angle between Normal and Photon Momentum
            G4double anglePhotonToNormal = 
                                          OldMomentum.angle(-theGlobalNormal);
            // Round it to closest integer
            G4int angleIncident = G4int(std::floor(180/pi*anglePhotonToNormal+0.5));

            // Take random angles THETA and PHI, 
            // and see if below Probability - if not - Redo
            do 
            {
                thetaIndex = G4RandFlat::shootInt(thetaIndexMax-1);
                phiIndex = G4RandFlat::shootInt(phiIndexMax-1);
                // Find probability with the new indeces from LUT
                AngularDistributionValue = OpticalSurface -> 
                   GetAngularDistributionValue(angleIncident,
                                               thetaIndex,
                                               phiIndex);
                // Loop checking, 13-Aug-2015, Peter Gumplinger
            } 
            while ( !G4BooleanRand(AngularDistributionValue) );

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
    } 
    while (NewMomentum * theGlobalNormal <= 0.0);
}

void InstrumentedG4OpBoundaryProcess::DielectricLUTDAVIS()
{
    G4int angindex, random, angleIncident;
    G4double ReflectivityValue, elevation, azimuth, EdotN;
    G4double anglePhotonToNormal;

    G4int LUTbin = OpticalSurface->GetLUTbins();
    G4double rand = G4UniformRand();
    do 
    {
        anglePhotonToNormal = OldMomentum.angle(-theGlobalNormal);
        angleIncident = G4int(std::floor(180/pi*anglePhotonToNormal+0.5));

        ReflectivityValue = OpticalSurface -> GetReflectivityLUTValue(angleIncident);

        if ( rand > ReflectivityValue ) 
        {
            if ( theEfficiency > 0 ) 
            {
                DoAbsorption();
                break;
            }
            else 
            {
                theStatus = Transmission;

                if (angleIncident <= 0.01) 
                {
                    NewMomentum = OldMomentum;
                    break;
                }

                do 
                {
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
        else 
        {
            theStatus = LobeReflection;
            if (angleIncident == 0) 
            {
                NewMomentum = -OldMomentum;
                break;
            }

            do 
            {
                random   = G4RandFlat::shootInt(1,LUTbin+1);
                angindex = (((random*2)-1))+(angleIncident-1)*LUTbin*2;

                azimuth   = OpticalSurface -> GetAngularDistributionValueLUT(angindex-1);
                elevation = OpticalSurface -> GetAngularDistributionValueLUT(angindex);
            } 
            while (elevation == 0 && azimuth == 0);

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
    } 
    while (NewMomentum * theGlobalNormal <= 0.0);
}

void InstrumentedG4OpBoundaryProcess::DielectricDichroic()
{
    // Calculate Angle between Normal and Photon Momentum
    G4double anglePhotonToNormal = OldMomentum.angle(-theGlobalNormal);

    // Round it to closest integer
    G4double angleIncident = std::floor(180/pi*anglePhotonToNormal+0.5);

    if (!DichroicVector) 
    {
        if (OpticalSurface) DichroicVector = OpticalSurface->GetDichroicVector();
    }

    if (DichroicVector) 
    {
        G4double wavelength = h_Planck*c_light/thePhotonMomentum;
        theTransmittance = DichroicVector->Value(wavelength/nm,angleIncident,idx,idy)*perCent; 

//      G4cout << "wavelength: " << std::floor(wavelength/nm) 
//                               << "nm" << G4endl;
//      G4cout << "Incident angle: " << angleIncident << "deg" << G4endl;
//      G4cout << "Transmittance: " 
//             << std::floor(theTransmittance/perCent) << "%" << G4endl;
    } 
    else 
    {
        G4ExceptionDescription ed;
        ed << " InstrumentedG4OpBoundaryProcess/DielectricDichroic(): "
           << " The dichroic surface has no G4Physics2DVector"
           << G4endl;
        G4Exception("InstrumentedG4OpBoundaryProcess::DielectricDichroic", "OpBoun03",
                    FatalException,ed,
                    "A dichroic surface must have an associated G4Physics2DVector");
    }

    if ( !G4BooleanRand(theTransmittance) ) 
    {   // Not transmitted, so reflect

        if ( theModel == glisur || theFinish == polished ) 
        {
            DoReflection();
        } 
        else 
        {
            ChooseReflection();
            if ( theStatus == LambertianReflection ) 
            {
                DoReflection();
            } 
            else if ( theStatus == BackScattering ) 
            {
                NewMomentum = -OldMomentum;
                NewPolarization = -OldPolarization;
            } 
            else 
            {
                G4double PdotN, EdotN;
                do 
                {
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
    } 
    else 
    {
        theStatus = Dichroic;
        NewMomentum = OldMomentum;
        NewPolarization = OldPolarization;
    }
}


/**
InstrumentedG4OpBoundaryProcess::DielectricDielectric
------------------------------------------------------

Sets::

   NewMomentum
   NewPolarization
   theStatus

**/


void InstrumentedG4OpBoundaryProcess::DielectricDielectric()
{
    G4bool Inside = false;
    G4bool Swap = false;

    G4bool SurfaceRoughnessCriterionPass = 1;
    if (theSurfaceRoughness != 0. && Rindex1 > Rindex2) 
    {
        G4double wavelength = h_Planck*c_light/thePhotonMomentum;
        G4double SurfaceRoughnessCriterion = std::exp(-std::pow((4*pi*theSurfaceRoughness*Rindex1*cost1/wavelength),2)); 
        SurfaceRoughnessCriterionPass = G4BooleanRand(SurfaceRoughnessCriterion); 
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

    do 
    {
        if (Through) 
        {
            Swap = !Swap;
            Through = false;
            theGlobalNormal = -theGlobalNormal;
            G4SwapPtr(Material1,Material2);
            G4SwapObj(&Rindex1,&Rindex2);
        }

        theFacetNormal = theFinish == polished ? theGlobalNormal : GetFacetNormal(OldMomentum,theGlobalNormal) ; 

        PdotN = OldMomentum * theFacetNormal;
        EdotN = OldPolarization * theFacetNormal;

        cost1 = - PdotN;
        if (std::abs(cost1) < 1.0-kCarTolerance)
        {
            sint1 = std::sqrt(1.-cost1*cost1);
            sint2 = sint1*Rindex1/Rindex2;     // *** Snell's Law ***
        }
        else 
        {
            sint1 = 0.0;
            sint2 = 0.0;
        }

#ifdef DEBUG_PIDX
        LOG_IF(LEVEL, pidx_dump)
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
            ;    
#endif

        if (sint2 >= 1.0) 
        {
            // Simulate total internal reflection
            if (Swap) Swap = !Swap;

            theStatus = TotalInternalReflection;
            if ( !SurfaceRoughnessCriterionPass ) theStatus = LambertianReflection; 

            if ( theModel == unified && theFinish != polished ) ChooseReflection(); 

            if ( theStatus == LambertianReflection ) 
            {
                DoReflection();
            }
            else if ( theStatus == BackScattering ) 
            {
                NewMomentum = -OldMomentum;
                NewPolarization = -OldPolarization;
            }
            else 
            {
                PdotN = OldMomentum * theFacetNormal;
                NewMomentum = OldMomentum - (2.*PdotN)*theFacetNormal;
                EdotN = OldPolarization * theFacetNormal;
                NewPolarization = -OldPolarization + (2.*EdotN)*theFacetNormal;
            }
        }
        else if (sint2 < 1.0) 
        {

            // Calculate amplitude for transmission (Q = P x N)

            if (cost1 > 0.0) 
            {
                cost2 =  std::sqrt(1.-sint2*sint2);
            }
            else 
            {
                cost2 = -std::sqrt(1.-sint2*sint2);
            }

            if (sint1 > 0.0)
            {
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
            else 
            {
#ifdef DEBUG_PIDX
                LOG_IF(LEVEL, pidx_dump)
                   << " pidx " << pidx 
                   << " NOT:sint1 > 0 : JACKSON NORMAL INCIDENCE "
                   ;
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
            LOG_IF(LEVEL, pidx_dump)
                 << " pidx " << pidx 
                 << " TransCoeff " << TransCoeff
                 ;
#endif
            if ( !G4BooleanRand_TransCoeff(TransCoeff) ) 
            {
                // Simulate reflection
                if (Swap) Swap = !Swap;

                theStatus = FresnelReflection;

                if ( !SurfaceRoughnessCriterionPass ) theStatus = LambertianReflection; 

                if ( theModel == unified && theFinish != polished ) ChooseReflection(); 

                if ( theStatus == LambertianReflection ) 
                {
                    DoReflection();
                }
                else if ( theStatus == BackScattering ) 
                {
                    NewMomentum = -OldMomentum;
                    NewPolarization = -OldPolarization;
                }
                else 
                {
                    PdotN = OldMomentum * theFacetNormal;
                    NewMomentum = OldMomentum - (2.*PdotN)*theFacetNormal;

                    if (sint1 > 0.0) 
                    {   // incident ray oblique

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
                    else 
                    {     // incident ray perpendicular

                        if (Rindex2 > Rindex1) 
                        {
                            NewPolarization = - OldPolarization;
                        }
                        else 
                        {
                            NewPolarization =   OldPolarization;
                        }
                    }
                }
            }
            else 
            {   // photon gets transmitted
                // Simulate transmission/refraction
#ifdef DEBUG_PIDX
                LOG_IF(LEVEL, pidx_dump) 
                    << " pidx " << pidx 
                    << " TRANSMIT "
                    ; 
#endif
                Inside = !Inside;
                Through = true;
                theStatus = FresnelRefraction;

                if (sint1 > 0.0) 
                {      // incident ray oblique

                    alpha = cost1 - cost2*(Rindex2/Rindex1);
                    NewMomentum = OldMomentum + alpha*theFacetNormal;
                    NewMomentum = NewMomentum.unit();
//                      PdotN = -cost2;
                    A_paral = NewMomentum.cross(A_trans);
                    A_paral = A_paral.unit();
                    E2_abs     = std::sqrt(E2_total);
                    C_parl     = E2_parl/E2_abs;
                    C_perp     = E2_perp/E2_abs;

                    NewPolarization = C_parl*A_paral + C_perp*A_trans;
                }
                else  
                {    // incident ray perpendicular
                    NewMomentum = OldMomentum;
                    NewPolarization = OldPolarization;
                }
#ifdef DEBUG_PIDX
                LOG_IF(LEVEL, pidx_dump)
                     << " pidx " << pidx 
                     << " TRANSMIT "
                     << " NewMom "
                     <<  "(" 
                     << " " << NewMomentum.x()
                     << " " << NewMomentum.y()
                     << " " << NewMomentum.z()
                     << ")" 
                     << " NewPol "
                     <<  "(" 
                     << " " << NewPolarization.x()
                     << " " << NewPolarization.y()
                     << " " << NewPolarization.z()
                     << ")" 
                     ;
                //
#endif
            }
        }

        OldMomentum = NewMomentum.unit();
        OldPolarization = NewPolarization.unit();

        if (theStatus == FresnelRefraction) 
        {
            Done = (NewMomentum * theGlobalNormal <= 0.0);
        } 
        else 
        {
            Done = (NewMomentum * theGlobalNormal >= -kCarTolerance);
        }

        // Loop checking, 13-Aug-2015, Peter Gumplinger
    } while (!Done);

    if (Inside && !Swap) 
    {
        if( theFinish == polishedbackpainted ||  theFinish == groundbackpainted ) 
        {

            G4double rand = G4UniformRand();
            if ( rand > theReflectivity ) 
            {
                if (rand > theReflectivity + theTransmittance) 
                {
                    DoAbsorption();
                } 
                else 
                {
                    theStatus = Transmission;
                    NewMomentum = OldMomentum;
                    NewPolarization = OldPolarization;
                }
            }
            else 
            {
                if (theStatus != FresnelRefraction )
                {
                    theGlobalNormal = -theGlobalNormal;
                }
                else 
                {
                    Swap = !Swap;
                    G4SwapPtr(Material1,Material2);
                    G4SwapObj(&Rindex1,&Rindex2);
                }
                if ( theFinish == groundbackpainted ) theStatus = LambertianReflection; 

                DoReflection();

                theGlobalNormal = -theGlobalNormal;
                OldMomentum = NewMomentum;

                goto leap;
            }
        }
    }
}


/**
InstrumentedG4OpBoundaryProcess::GetIncidentAngle
---------------------------------------------------

SCB: HUH, both magP and magN should be 1 ? 
SCB; also why bother to acos, what you really need is cos_angle and sin_angle

**/

G4double InstrumentedG4OpBoundaryProcess::GetIncidentAngle() 
{
    G4double PdotN = OldMomentum * theFacetNormal;
    G4double magP= OldMomentum.mag();
    G4double magN= theFacetNormal.mag();

    G4double incidentangle = pi - std::acos(PdotN/(magP*magN));

    return incidentangle;
}

/**
InstrumentedG4OpBoundaryProcess::GetReflectivity
--------------------------------------------------

Consumes randoms in do-while to set iTE, iTM

**/

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

    G4MaterialPropertiesTable* aMaterialPropertiesTable = Material1->GetMaterialPropertiesTable(); 
    G4MaterialPropertyVector* aPropertyPointerR = aMaterialPropertiesTable->GetProperty(kREALRINDEX); 
    G4MaterialPropertyVector* aPropertyPointerI = aMaterialPropertiesTable->GetProperty(kIMAGINARYRINDEX); 

    if (aPropertyPointerR && aPropertyPointerI) 
    {
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

/**
InstrumentedG4OpBoundaryProcess::CalculateReflectivity
---------------------------------------------------------

UGLY: 

* depends on PropertyPointer1, PropertyPointer2 for Real and Imag Rindex
* sets theReflectivity 

**/

void InstrumentedG4OpBoundaryProcess::CalculateReflectivity()
{
    G4double RealRindex = PropertyPointer1->Value(thePhotonMomentum); 
    G4double ImaginaryRindex = PropertyPointer2->Value(thePhotonMomentum); 

    theFacetNormal = theFinish == ground ? GetFacetNormal(OldMomentum, theGlobalNormal) : theGlobalNormal ;

    G4double PdotN = OldMomentum * theFacetNormal;
    cost1 = -PdotN;
    sint1 = (std::abs(cost1) < 1.0 - kCarTolerance) ? std::sqrt(1. - cost1*cost1) : 0.0 ;

    G4ThreeVector A_trans, A_paral, E1pp, E1pl;
    G4double E1_perp, E1_parl;

    if (sint1 > 0.0 ) 
    {
        A_trans = OldMomentum.cross(theFacetNormal);
        A_trans = A_trans.unit();
        E1_perp = OldPolarization * A_trans;
        E1pp    = E1_perp * A_trans;
        E1pl    = OldPolarization - E1pp;
        E1_parl = E1pl.mag();
    }
    else 
    {
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

    theReflectivity = GetReflectivity(E1_perp, E1_parl, incidentangle, RealRindex, ImaginaryRindex); 
}


/**
InstrumentedG4OpBoundaryProcess::DoAbsorption
----------------------------------------------

Sets: 

* theStatus to Detection/Absorption depending on one random and theEfficiency 
* sets NewMomentum, NewPolarization to the Old 
* aParticleChange 

**/

void InstrumentedG4OpBoundaryProcess::DoAbsorption()
{
    LOG(LEVEL) 
        << " PostStepDoIt_count " << PostStepDoIt_count
        << " theEfficiency " << theEfficiency
        ;

    bool detect = G4BooleanRand_theEfficiency(theEfficiency) ; 
    theStatus = detect ? Detection : Absorption ; 

    NewMomentum = OldMomentum;
    NewPolarization = OldPolarization;

    aParticleChange.ProposeLocalEnergyDeposit(detect ? thePhotonMomentum : 0.0);
    aParticleChange.ProposeTrackStatus(fStopAndKill);
}


/**
InstrumentedG4OpBoundaryProcess::DoReflection
-----------------------------------------------

Sets:: 

   NewMomentum
   NewPolarization
   theFacetNormal
   theStatus (unless already LambertianReflection)


theStatus:LambertianReflection
    NewMomentum Lambertian sampled around theGlobalNormal, changes theFacetNormal  

theFinish:ground
    sets theStatus:LobeReflection



           B
          /|\
         / | \
        /  |  \
       /   |   \
      + - -+- - +      
       \   |   /
       OLD |  NEW
         \ | / 
          \|/
  ---------A------------

    A->B = NEW - OLD

**/

void InstrumentedG4OpBoundaryProcess::DoReflection()
{
    if( theStatus == LambertianReflection ) 
    {
        NewMomentum = U4LambertianRand(theGlobalNormal);
        theFacetNormal = (NewMomentum - OldMomentum).unit();
    }
    else if( theFinish == ground ) 
    {
        theStatus = LobeReflection;
        if (PropertyPointer1 && PropertyPointer2)
        {
        } 
        else 
        {
            theFacetNormal = GetFacetNormal(OldMomentum,theGlobalNormal); 
        }
        G4double PdotN = OldMomentum * theFacetNormal;
        NewMomentum = OldMomentum - (2.*PdotN)*theFacetNormal;
   }
   else 
   {
       theStatus = SpikeReflection;
       theFacetNormal = theGlobalNormal;
       G4double PdotN = OldMomentum * theFacetNormal;
       NewMomentum = OldMomentum - (2.*PdotN)*theFacetNormal;
   }
   G4double EdotN = OldPolarization * theFacetNormal;
   NewPolarization = -OldPolarization + (2.*EdotN)*theFacetNormal;
}


/**
InstrumentedG4OpBoundaryProcess::ChooseReflection
--------------------------------------------------

Sets theStatus to SpikeReflection/LobeReflection/BackScattering/LambertianReflection
depending on one rand and prob_ss/prob_sl/prob_bs. 

But prob_ss/prob_sl/prob_bs all default to zero and require unified model and appropriate 
properties in OpticalSurface.mpt to change from zero. So this will typically return LambertianReflection

For SpikeReflection theFacetNormal is set to theGlobalNormal
**/

void InstrumentedG4OpBoundaryProcess::ChooseReflection()
{
    G4double rand = G4UniformRand();
#ifdef DEBUG_TAG
    SEvt::AddTag( U4Stack_ChooseReflection, rand ); 
#endif

    if ( rand >= 0.0 && rand < prob_ss ) 
    {
        theStatus = SpikeReflection;
        theFacetNormal = theGlobalNormal;
    }
    else if ( rand >= prob_ss && rand <= prob_ss+prob_sl) 
    {
        theStatus = LobeReflection;
    }
    else if ( rand > prob_ss+prob_sl && rand < prob_ss+prob_sl+prob_bs ) 
    {
        theStatus = BackScattering;
    }
    else 
    {
        theStatus = LambertianReflection;
    }

    LOG(LEVEL) 
        << " theStatus " << U4OpBoundaryProcessStatus::Name(theStatus) 
        << " prob_ss " << prob_ss 
        << " prob_sl " << prob_sl 
        << " prob_bs " << prob_bs 
        ; 
}




G4bool InstrumentedG4OpBoundaryProcess::IsApplicable(const G4ParticleDefinition& aParticleType)
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


G4bool InstrumentedG4OpBoundaryProcess::G4BooleanRand(const G4double prob) const
{
    G4double u = G4UniformRand() ; 
    G4bool ret = u < prob  ; 
#ifdef DEBUG_PIDX
    LOG_IF(LEVEL, pidx_dump)
         << " pidx " << pidx
         << " prob " << prob 
         << " ret " << ret 
         ;   
#endif
    return ret ; 
}

G4bool InstrumentedG4OpBoundaryProcess::G4BooleanRand_TransCoeff(const G4double prob) const
{
    G4double u = G4UniformRand() ; 
    G4bool ret = u < prob  ; 

    LOG(LEVEL) 
        << U4UniformRand::Desc(u, SEvt::UU)
        << " TransCoeff " << prob 
        << " DECISION " << ( ret ? "T" : "R" ) 
        ; 
#ifdef DEBUG_TAG
    SEvt::AddTag(U4Stack_BoundaryDiDiTransCoeff, u ); 
#endif   
#ifdef DEBUG_PIDX
    LOG_IF(LEVEL, pidx_dump)
         << " pidx " << pidx
         << " prob " << prob 
         << " ret " << ret 
         ;   
#endif
    return ret ; 
}

G4bool InstrumentedG4OpBoundaryProcess::G4BooleanRand_theEfficiency(const G4double prob) const
{
    G4double u = G4UniformRand() ; 
    G4bool ret = u < prob  ; 
#ifdef DEBUG_TAG
    SEvt::AddTag(U4Stack_AbsorptionEffDetect, u ); 
#endif   
#ifdef DEBUG_PIDX
    LOG_IF(LEVEL, pidx_dump)
         << " pidx " << pidx
         << " prob " << prob 
         << " ret " << ret 
         ;   
#endif
    return ret ; 
}

G4bool InstrumentedG4OpBoundaryProcess::G4BooleanRand_theReflectivity(const G4double prob) const
{
    G4double u = G4UniformRand() ; 
    G4bool ret = u < prob ; 
#ifdef DEBUG_PIDX
    LOG_IF(LEVEL, pidx_dump)
        << " pidx " << pidx 
        << " prob " << prob
        << " u " << u 
        << " ret " << ret
        ;

#endif
    return ret  ; 
}


G4bool InstrumentedG4OpBoundaryProcess::InvokeSD(const G4Step* pStep)
{
    G4Step aStep = *pStep;
    aStep.AddTotalEnergyDeposit(thePhotonMomentum);
    G4VSensitiveDetector* sd = aStep.GetPostStepPoint()->GetSensitiveDetector();
    return sd ? sd->Hit(&aStep) : false ;
}

G4double InstrumentedG4OpBoundaryProcess::GetMeanFreePath(const G4Track& , G4double , G4ForceCondition* condition) 
{
    *condition = Forced;
    return DBL_MAX;
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


