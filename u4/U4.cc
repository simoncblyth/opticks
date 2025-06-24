#include <iomanip>
#include <iostream>
#include <cassert>
#include <csignal>
#include <cstdlib>

#include "G4VParticleChange.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "G4Track.hh"
#include "G4OpticalPhoton.hh"
#include "G4Event.hh"

#include "SEvt.hh"
#include "scuda.h"
#include "squad.h"
#include "sphoton.h"
#include "OpticksGenstep.h"

#include "SPath.hh"
#include "SStr.hh"
#include "NP.hh"
#include "SLOG.hh"
#include "sscint.h"
#include "scerenkov.h"
#include "sgs.h"

#ifdef WITH_CUSTOM4
#include "C4GS.h"
#include "C4Pho.h"
#include "C4TrackInfo.h"
#else
#include "spho.h"
#include "STrackInfo.h"
#endif

#include "U4.hh"

const plog::Severity U4::LEVEL = SLOG::EnvLevel("U4", "DEBUG");


/**
U4 private state
--------------------

Here are holding the state of the genstep collection in translation-unit-local static variables.

An alternative more o.o. approach would be to use a U4Private/U4Impl struct
that a U4 instance holds a pointer to and passes along messages to.
That is like the PImpl pattern : pointer to implementation.

* https://www.geeksforgeeks.org/pimpl-idiom-in-c-with-examples/
* https://www.cppstories.com/2018/01/pimpl/

HMM: perhapa this state belongs better within SEvt together with the full gensteps ?

**/


#ifdef WITH_CUSTOM4
static C4GS gs = {} ;            // updated by eg U4::CollectGenstep_DsG4Scintillation_r4695 prior to each photon generation loop
static C4Pho ancestor = {} ;     // updated by U4::GenPhotonAncestor prior to the photon generation loop(s)
static C4Pho pho = {} ;          // updated by U4::GenPhotonBegin at start of photon generation loop
static C4Pho secondary = {} ;    // updated by U4::GenPhotonEnd   at end of photon generation loop
#else
static sgs gs = {} ;            // updated by eg U4::CollectGenstep_DsG4Scintillation_r4695 prior to each photon generation loop
static spho ancestor = {} ;     // updated by U4::GenPhotonAncestor prior to the photon generation loop(s)
static spho pho = {} ;          // updated by U4::GenPhotonBegin at start of photon generation loop
static spho secondary = {} ;    // updated by U4::GenPhotonEnd   at end of photon generation loop
#endif

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
    const G4Material* aMaterial = aTrack->GetMaterial();

    quad6 _gs ;
    _gs.zero() ;

    sscint* gs = (sscint*)(&_gs) ;   // warning: dereferencing type-punned pointer will break strict-aliasing rules

    gs->gentype = OpticksGenstep_DsG4Scintillation_r4695 ;
    gs->trackid = aTrack->GetTrackID() ;
    gs->matline = aMaterial->GetIndex() + SEvt::G4_INDEX_OFFSET ;  // offset signals that a mapping must be done in SEvt::setGenstep
    gs->numphoton = numPhotons ;

    // note that gs->matline is not currently used for scintillation,
    // but done here as check of SEvt::addGenstep mtindex to mtline mapping

    gs->pos.x = x0.x() ;
    gs->pos.y = x0.y() ;
    gs->pos.z = x0.z() ;
    gs->time = t0 ;

    gs->DeltaPosition.x = deltaPosition.x() ;
    gs->DeltaPosition.y = deltaPosition.y() ;
    gs->DeltaPosition.z = deltaPosition.z() ;
    gs->step_length = aStep->GetStepLength() ;

    gs->code = aParticle->GetDefinition()->GetPDGEncoding() ;
    gs->charge = aParticle->GetDefinition()->GetPDGCharge() ;
    gs->weight = aTrack->GetWeight() ;
    gs->meanVelocity = meanVelocity ;

    gs->scnt = scnt ;
    gs->f41 = 0.f ;
    gs->f42 = 0.f ;
    gs->f43 = 0.f ;

    gs->ScintillationTime = ScintillationTime ;
    gs->f51 = 0.f ;
    gs->f52 = 0.f ;
    gs->f53 = 0.f ;

    return _gs ;
}


const char* U4::CollectGenstep_DsG4Scintillation_r4695_DISABLE = "U4__CollectGenstep_DsG4Scintillation_r4695_DISABLE" ;
const char* U4::CollectGenstep_DsG4Scintillation_r4695_ZEROPHO = "U4__CollectGenstep_DsG4Scintillation_r4695_ZEROPHO" ;

void U4::CollectGenstep_DsG4Scintillation_r4695(
         const G4Track* aTrack,
         const G4Step* aStep,
         G4int    numPhotons,
         G4int    scnt,
         G4double ScintillationTime
    )
{
    if(getenv(CollectGenstep_DsG4Scintillation_r4695_DISABLE))
    {
        LOG(error) << CollectGenstep_DsG4Scintillation_r4695_DISABLE ;
        return ;
    }
    if(getenv(CollectGenstep_DsG4Scintillation_r4695_ZEROPHO))
    {
        LOG(error) << CollectGenstep_DsG4Scintillation_r4695_ZEROPHO ;
        numPhotons = 0 ;
    }



    quad6 gs_ = MakeGenstep_DsG4Scintillation_r4695( aTrack, aStep, numPhotons, scnt, ScintillationTime);

#ifdef WITH_CUSTOM4
    sgs _gs = SEvt::AddGenstep(gs_);    // returns sgs struct which is a simple 4 int label
    gs = C4GS::Make(_gs.index, _gs.photons, _gs.offset, _gs.gentype );
#else
    gs = SEvt::AddGenstep(gs_);    // returns sgs struct which is a simple 4 int label
#endif
    // gs is private static genstep label

    //if(dump) std::cout << "U4::CollectGenstep_DsG4Scintillation_r4695 " << gs.desc() << std::endl ;
    LOG(LEVEL) << gs.desc();
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

    G4double Wmin_nm = h_Planck*c_light/pmax/nm ;
    G4double Wmax_nm = h_Planck*c_light/pmin/nm ;

    const G4Material* aMaterial = aTrack->GetMaterial();

    quad6 _gs ;
    _gs.zero() ;

    scerenkov* gs = (scerenkov*)(&_gs) ;

    gs->gentype = OpticksGenstep_G4Cerenkov_modified ;
    gs->trackid = aTrack->GetTrackID() ;
    gs->matline = aMaterial->GetIndex() + SEvt::G4_INDEX_OFFSET ;  // offset signals that a mapping must be done in SEvt::setGenstep
    gs->numphoton = numPhotons ;

    gs->pos.x = x0.x() ;
    gs->pos.y = x0.y() ;
    gs->pos.z = x0.z() ;
    gs->time = t0 ;

    gs->DeltaPosition.x = deltaPosition.x() ;
    gs->DeltaPosition.y = deltaPosition.y() ;
    gs->DeltaPosition.z = deltaPosition.z() ;
    gs->step_length = aStep->GetStepLength() ;

    gs->code = aParticle->GetDefinition()->GetPDGEncoding() ;
    gs->charge = aParticle->GetDefinition()->GetPDGCharge() ;
    gs->weight = aTrack->GetWeight() ;
    gs->preVelocity = pPreStepPoint->GetVelocity() ;

    gs->BetaInverse = betaInverse ;
    gs->Wmin = Wmin_nm ;
    gs->Wmax = Wmax_nm   ;
    gs->maxCos = maxCos ;

    gs->maxSin2 = maxSin2 ;
    gs->MeanNumberOfPhotons1 = meanNumberOfPhotons1 ;
    gs->MeanNumberOfPhotons2 = meanNumberOfPhotons2 ;
    gs->postVelocity = pPostStepPoint->GetVelocity() ;

    return _gs ;
}

const char* U4::CollectGenstep_G4Cerenkov_modified_DISABLE = "U4__CollectGenstep_G4Cerenkov_modified_DISABLE" ;
const char* U4::CollectGenstep_G4Cerenkov_modified_ZEROPHO = "U4__CollectGenstep_G4Cerenkov_modified_ZEROPHO" ;

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

    if(getenv(CollectGenstep_G4Cerenkov_modified_DISABLE))
    {
        LOG(error) << CollectGenstep_G4Cerenkov_modified_DISABLE ;
        return ;
    }
    if(getenv(CollectGenstep_G4Cerenkov_modified_ZEROPHO))
    {
        LOG(error) << CollectGenstep_G4Cerenkov_modified_ZEROPHO ;
        numPhotons = 0 ;
    }




    quad6 gs_ = MakeGenstep_G4Cerenkov_modified( aTrack, aStep, numPhotons, betaInverse, pmin, pmax, maxCos, maxSin2, meanNumberOfPhotons1, meanNumberOfPhotons2 );

#ifdef WITH_CUSTOM4
    sgs _gs = SEvt::AddGenstep(gs_);    // returns sgs struct which is a simple 4 int label
    gs = C4GS::Make(_gs.index, _gs.photons, _gs.offset , _gs.gentype );
#else
    gs = SEvt::AddGenstep(gs_);    // returns sgs struct which is a simple 4 int label
#endif
    // gs is primate static genstep label
    // TODO: avoid the duplication betweek C and S with common SetGenstep private method

    if(dump) std::cout << "U4::CollectGenstep_G4Cerenkov_modified " << gs.desc() << std::endl ;
    LOG(LEVEL) << gs.desc();
}







/**
U4::GenPhotonAncestor
----------------------

NB this is called prior to generation loops to get the ancestor spho.h label

This label is needed for BOTH Scintillation and Cerenkov in order for photon G4Track
labelling done by U4::GenPhotonEnd to work.

When the track has no user info the ancestor is set to spho::Placeholder label {-1,-1,-1,-1}.

If the call to U4::GenPhotonAncestor is omitted from the Scintillation OR Cerenkov
PostStepDoIt then the ancestor will default to {0,0,0,0} : that will cause
unexpected labels.

**/

void U4::GenPhotonAncestor( const G4Track* aTrack )
{
#ifdef WITH_CUSTOM4
    ancestor = C4TrackInfo<C4Pho>::Get(aTrack) ;
#else
    ancestor = STrackInfo<spho>::Get(aTrack) ;
#endif
    if(dump) std::cout << "U4::GenPhotonAncestor " << ancestor.desc() << std::endl ;
    LOG(LEVEL) << ancestor.desc() ;
}

/**
U4::GenPhotonBegin
-------------------

This is called from head of Scintillation and Cerenkov generation loops.
It updates the private "pho" spho.h label which carries
genstep index, photon index within genstep, photon identity, reemission index

**/

void U4::GenPhotonBegin( int genloop_idx )
{
    assert(genloop_idx > -1);
    pho = gs.MakePho(genloop_idx, ancestor);

    int align_id = ancestor.isPlaceholder() ? gs.offset + genloop_idx : ancestor.id ;

    bool align_id_expect = pho.id == align_id ;
    assert( align_id_expect );
    if(!align_id_expect) std::raise(SIGINT);

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

This is called from tail of Scintillation and Cerenkov generation loops
following instanciation of the secondary track.

Sets spho label into the secondary track using U4PhotonInfo::Set

**/

void U4::GenPhotonEnd( int genloop_idx, G4Track* aSecondaryTrack )
{
    assert(genloop_idx > -1);
    secondary = gs.MakePho(genloop_idx, ancestor) ;
    assert( secondary.isIdentical(pho) );  // make sure paired U4::GenPhotonBegin and U4::GenPhotonEnd

#ifdef DEBUG
    if(dump) std::cout << "U4::GenPhotonEnd " << secondary.desc() << std::endl ;
#endif

#ifdef WITH_CUSTOM4
    C4TrackInfo<C4Pho>::Set(aSecondaryTrack, secondary );
#else
    STrackInfo<spho>::Set(aSecondaryTrack, secondary );
#endif

}



/**
U4::CollectOpticalSecondaries
------------------------------

Optional and as yet incomplete conversion of G4VParticleChange into NP array of photons

**/

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

        // position ?

        sp.mom.x = pmom.x();
        sp.mom.y = pmom.y();
        sp.mom.z = pmom.z();

        sp.pol.x = ppol.x();
        sp.pol.y = ppol.y();
        sp.pol.z = ppol.z();
    }
    return p ;
}


void U4::GenPhotonSecondaries( const G4Track* , const G4VParticleChange* )
{
    // do nothing
}


