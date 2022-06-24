#include <iomanip>
#include <iostream>
#include <cassert>

#include "G4VParticleChange.hh"
#include "G4Track.hh"
#include "G4OpticalPhoton.hh"
#include "G4Event.hh"

#include "SEvt.hh"
#include "scuda.h"
#include "squad.h"
#include "sphoton.h"
#include "sgs.h"
#include "spho.h"
#include "sscint.h"
#include "OpticksGenstep.h"

#include "SPath.hh"
#include "SStr.hh"
#include "NP.hh"
#include "PLOG.hh"
#include "sscint.h"
#include "scerenkov.h"

#include "U4PhotonInfo.h"
#include "U4.hh" 

const plog::Severity U4::LEVEL = PLOG::EnvLevel("U4", "DEBUG"); 


/**
U4 private state
--------------------

Here are holding the state of the genstep collection in translation-unit-local static variables. 

An alternative more o.o. approach would be to use a U4Private/U4Impl struct
that a U4 instance holds a pointer to and passes along messages to.  
That is like the PImpl pattern : pointer to implementation.

* https://www.geeksforgeeks.org/pimpl-idiom-in-c-with-examples/
* https://www.cppstories.com/2018/01/pimpl/

**/

// HMM: perhapa this state belongs better within SEvt together with the full gensteps ?

static spho ancestor = {} ;     // updated by U4::GenPhotonAncestor prior to the photon generation loop(s)
static sgs gs = {} ;            // updated by eg U4::CollectGenstep_DsG4Scintillation_r4695 prior to each photon generation loop 
static spho pho = {} ;          // updated by U4::GenPhotonBegin at start of photon generation loop
static spho secondary = {} ;    // updated by U4::GenPhotonEnd   at end of photon generation loop 

static bool dump = false ; 

/**
MakeGenstep... Hidden Functions only usable from this translation unit 
------------------------------------------------------------------------

Using hidden static non-member functions allows keeping Opticks types like quad6 out of U4 header

**/

static quad6 MakeGenstep_DsG4Scintillation_r4695( 
     const G4Track* aTrack,
     const G4Step* aStep,
     G4int    numPhotons,
     G4int    scnt,        
     G4double ScintillationTime
    )
{
    G4StepPoint* pPreStepPoint  = aStep->GetPreStepPoint();
    G4StepPoint* pPostStepPoint = aStep->GetPostStepPoint();

    G4ThreeVector x0 = pPreStepPoint->GetPosition();
    G4double      t0 = pPreStepPoint->GetGlobalTime();
    G4ThreeVector deltaPosition = aStep->GetDeltaPosition() ;
    G4double meanVelocity = (pPreStepPoint->GetVelocity()+pPostStepPoint->GetVelocity())/2. ;

    const G4DynamicParticle* aParticle = aTrack->GetDynamicParticle();
    //const G4Material* aMaterial = aTrack->GetMaterial();

    quad6 _gs ;
    _gs.zero() ; 
    
    sscint& gs = (sscint&)_gs ; 

    gs.gentype = OpticksGenstep_DsG4Scintillation_r4695 ;
    gs.trackid = aTrack->GetTrackID() ;
    gs.matline = 0u ; //  aMaterial->GetIndex()   // not used for scintillation
    gs.numphoton = numPhotons ;  

    gs.pos.x = x0.x() ; 
    gs.pos.y = x0.y() ; 
    gs.pos.z = x0.z() ; 
    gs.time = t0 ; 

    gs.DeltaPosition.x = deltaPosition.x() ; 
    gs.DeltaPosition.y = deltaPosition.y() ; 
    gs.DeltaPosition.z = deltaPosition.z() ; 
    gs.step_length = aStep->GetStepLength() ;

    gs.code = aParticle->GetDefinition()->GetPDGEncoding() ;
    gs.charge = aParticle->GetDefinition()->GetPDGCharge() ;
    gs.weight = aTrack->GetWeight() ;
    gs.meanVelocity = meanVelocity ; 

    gs.scnt = scnt ; 
    gs.f41 = 0.f ;  
    gs.f42 = 0.f ;  
    gs.f43 = 0.f ; 

    gs.ScintillationTime = ScintillationTime ;
    gs.f51 = 0.f ;
    gs.f52 = 0.f ;
    gs.f53 = 0.f ;

    return _gs ; 
}

void U4::CollectGenstep_DsG4Scintillation_r4695( 
         const G4Track* aTrack,
         const G4Step* aStep,
         G4int    numPhotons,
         G4int    scnt,        
         G4double ScintillationTime
    )
{
    quad6 gs_ = MakeGenstep_DsG4Scintillation_r4695( aTrack, aStep, numPhotons, scnt, ScintillationTime);
    gs = SEvt::AddGenstep(gs_);    // returns sgs struct which is a simple 4 int label 
    //if(dump) std::cout << "U4::CollectGenstep_DsG4Scintillation_r4695 " << gs.desc() << std::endl ; 
}



static quad6 MakeGenstep_G4Cerenkov_modified( 
    const G4Track* aTrack,
    const G4Step* aStep,
    G4int    numPhotons,
    G4double    betaInverse,
    G4double    pmin,
    G4double    pmax,
    G4double    maxCos,

    G4double    maxSin2,
    G4double    meanNumberOfPhotons1,
    G4double    meanNumberOfPhotons2
    )    
{
    G4StepPoint* pPreStepPoint  = aStep->GetPreStepPoint();
    G4StepPoint* pPostStepPoint = aStep->GetPostStepPoint();

    G4ThreeVector x0 = pPreStepPoint->GetPosition();
    G4double      t0 = pPreStepPoint->GetGlobalTime();
    G4ThreeVector deltaPosition = aStep->GetDeltaPosition() ;

    const G4DynamicParticle* aParticle = aTrack->GetDynamicParticle();

    quad6 _gs ;
    _gs.zero() ; 
    
    scerenkov& gs = (scerenkov&)_gs ; 

    gs.gentype = OpticksGenstep_G4Cerenkov_modified ;  
    gs.trackid = aTrack->GetTrackID() ;
    gs.matline = 0u ; //  aMaterial->GetIndex()  
    gs.numphoton = numPhotons ;  

    gs.pos.x = x0.x() ; 
    gs.pos.y = x0.y() ; 
    gs.pos.z = x0.z() ; 
    gs.time = t0 ; 

    gs.DeltaPosition.x = deltaPosition.x() ; 
    gs.DeltaPosition.y = deltaPosition.y() ; 
    gs.DeltaPosition.z = deltaPosition.z() ; 
    gs.step_length = aStep->GetStepLength() ;

    gs.code = aParticle->GetDefinition()->GetPDGEncoding() ;
    gs.charge = aParticle->GetDefinition()->GetPDGCharge() ;
    gs.weight = aTrack->GetWeight() ;
    gs.preVelocity = pPreStepPoint->GetVelocity() ;

    gs.BetaInverse = betaInverse ; 
    gs.Wmin = 0.f ;  
    gs.Wmax= 0.f ;  
    gs.maxCos = maxCos ; 

    gs.maxSin2 = maxSin2 ;
    gs.MeanNumberOfPhotons1 = meanNumberOfPhotons1 ;
    gs.MeanNumberOfPhotons2 = meanNumberOfPhotons2 ;
    gs.postVelocity = pPostStepPoint->GetVelocity() ;

    return _gs ; 
}


void U4::CollectGenstep_G4Cerenkov_modified( 
    const G4Track* aTrack,
    const G4Step* aStep,
    G4int    numPhotons,
    G4double    betaInverse,
    G4double    pmin,
    G4double    pmax,
    G4double    maxCos,
    G4double    maxSin2,
    G4double    meanNumberOfPhotons1,
    G4double    meanNumberOfPhotons2)
{
    quad6 gs_ = MakeGenstep_G4Cerenkov_modified( aTrack, aStep, numPhotons, betaInverse, pmin, pmax, maxCos, maxSin2, meanNumberOfPhotons1, meanNumberOfPhotons2 );
    gs = SEvt::AddGenstep(gs_);    // returns sgs struct which is a simple 4 int label 
    if(dump) std::cout << "U4::CollectGenstep_G4Cerenkov_modified " << gs.desc() << std::endl ; 
}







/**
U4::GenPhotonAncestor
----------------------

NB calling this prior to generation loops to get the ancestor 
is needed for BOTH Scintillation and Cerenkov in order for photon G4Track 
labelling done by U4::GenPhotonEnd to work. 

**/

void U4::GenPhotonAncestor( const G4Track* aTrack )
{
    ancestor = U4PhotonInfo::Get(aTrack) ; 
    if(dump) std::cout << "U4::GenPhotonAncestor " << ancestor.desc() << std::endl ;  
}

/**
U4::GenPhotonBegin
-------------------

**/

void U4::GenPhotonBegin( int genloop_idx )
{
    assert(genloop_idx > -1); 
    pho = gs.MakePho(genloop_idx, ancestor); 

    int align_id = ancestor.isPlaceholder() ? gs.offset + genloop_idx : ancestor.id ; 
    assert( pho.id == align_id );     

#ifdef DEBUG
    if(dump) std::cout 
        << "U4::GenPhotonBegin"
        << " genloop_idx " << std::setw(6) << genloop_idx 
        << " gs.offset " << std::setw(6) << gs.offset 
        << " pho.id " << std::setw(6) << pho.id
        << std::endl 
        ; 
#endif
}

/**
U4::GenPhotonEnd
------------------

Sets spho label into the secondary track using U4PhotonInfo::Set

**/

void U4::GenPhotonEnd( int genloop_idx, G4Track* aSecondaryTrack )
{
    assert(genloop_idx > -1); 
    secondary = gs.MakePho(genloop_idx, ancestor) ; 

    assert( secondary.isIdentical(pho) ); 

    //std::cout << "U4::GenPhotonEnd " << secondary.desc() << std::endl ; 
#ifdef DEBUG
    if(dump) std::cout << "U4::GenPhotonEnd " << secondary.desc() << std::endl ; 
#endif

    U4PhotonInfo::Set(aSecondaryTrack, secondary ); 
}

void U4::GenPhotonSecondaries( const G4Track* aTrack, const G4VParticleChange* change )
{
    G4int numSecondaries = change->GetNumberOfSecondaries() ; 
    if(dump) std::cout << "U4::GenPhotonSecondaries  numSecondaries " << numSecondaries << std::endl ; 

    /*
    //TODO: reinstate some form of consistency check 

    // HMM: only works for 1st genstep of event perhaps ?

    int numphoton = SEvt::GetNumPhoton() ; 
    bool consistent = numphoton > -1 && numphoton - 1  == pho.id ;  
    //if( dump || !consistent )
    { 
        std::cout << " consistent " << consistent << std::endl ; 
        std::cout << " SEvt::GetNumPhoton " << numphoton << std::endl ; 
        std::cout << " pho " << pho.desc() << std::endl ; 
        std::cout << " gs " << gs.desc() << std::endl ; 
    }
    assert(consistent); 
    */
}

NP* U4::CollectOpticalSecondaries(const G4VParticleChange* pc )
{
    G4int num = pc->GetNumberOfSecondaries();

    std::cout << "U4::CollectOpticalSecondaries num " << num << std::endl ; 

    NP* p = NP::Make<float>(num, 4, 4); 
    sphoton* pp = (sphoton*)p->bytes() ; 

    for(int i=0 ; i < num ; i++)
    {   
        G4Track* track =  pc->GetSecondary(i) ;
        assert( track->GetParticleDefinition() == G4OpticalPhoton::Definition() );
        const G4DynamicParticle* ph = track->GetDynamicParticle() ;
        const G4ThreeVector& pmom = ph->GetMomentumDirection() ;
        const G4ThreeVector& ppol = ph->GetPolarization() ;
        sphoton& sp = pp[i] ; 

        sp.mom.x = pmom.x(); 
        sp.mom.y = pmom.y(); 
        sp.mom.z = pmom.z(); 

        sp.pol.x = ppol.x(); 
        sp.pol.y = ppol.y(); 
        sp.pol.z = ppol.z(); 
    }
    return p ; 
} 


