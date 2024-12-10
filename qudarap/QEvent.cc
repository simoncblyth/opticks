#include <cuda_runtime.h>
#include <sstream>
#include <csignal>



#include "SEvt.hh"

#include "scuda.h"
#include "squad.h"
#include "sphoton.h"
#include "sslice.h"

#ifndef PRODUCTION
#include "srec.h"
#include "sseq.h"
#include "stag.h"
#endif

#include "sevent.h"
#include "salloc.h"
#include "sstamp.h"
#include "ssys.h"

#include "sqat4.h"
#include "stran.h"
#include "SU.hh"

#include "SComp.h"
#include "SGenstep.h"
#include "SEvent.hh"
#include "SEvt.hh"
#include "SEventConfig.hh"
#include "NP.hh"
#include "SLOG.hh"

#include "OpticksGenstep.h"

#include "QEvent.hh"
#include "QBuf.hh"
#include "QBuf.hh"
#include "QU.hh"


template struct QBuf<quad6> ; 

bool QEvent::LIFECYCLE = ssys::getenvbool(QEvent__LIFECYCLE) ;

const plog::Severity QEvent::LEVEL = SLOG::EnvLevel("QEvent", "DEBUG"); 
QEvent* QEvent::INSTANCE = nullptr ; 
QEvent* QEvent::Get(){ return INSTANCE ; }

const bool QEvent::SEvt_NPFold_VERBOSE  = ssys::getenvbool("QEvent__SEvt_NPFold_VERBOSE") ;

std::string QEvent::Desc() // static
{
    std::stringstream ss ; 
    ss << "QEvent::Desc" << std::endl 
       << " QEvent__SEvt_NPFold_VERBOSE     : " << ( SEvt_NPFold_VERBOSE     ? "YES" : "NO " ) << std::endl
       ;

    std::string str = ss.str(); 
    return str ; 
}



sevent* QEvent::getDevicePtr() const
{
    return d_evt ; 
}

/**
QEvent::QEvent
----------------

Canonical QEvent instance resides within QSim and is instanciated by QSim::QSim.
Instanciation allocates device buffers with sizes configured by SEventConfig

* As selector is only needed CPU side it is not down in sevent.h 
  but it could be in SEvt.hh 

Holds:

* SEvt.hh:sev 
* sevent.h:evt
* sevent.h:d_evt
* NP.hh:gs
* NP.hh:input_photon


Q: Where is the SEvt::EGPU instanciated ?

**/

QEvent::QEvent()
    :
    sev(SEvt::Get_EGPU()),
    selector(sev ? sev->selector : nullptr),
    evt(sev ? sev->evt : nullptr),
    d_evt(QU::device_alloc<sevent>(1,"QEvent::QEvent/sevent")),
    gs(nullptr),
    gss(nullptr),
    input_photon(nullptr),
    upload_count(0)
{
    LOG(LEVEL); 
    LOG_IF(info, LIFECYCLE) ; 
    INSTANCE = this ; 
    init(); 
}

/**
QEvent::init
--------------

Only configures limits, no allocation yet. Allocation happens in QEvent::setGenstep QEvent::setNumPhoton

HMM: hostside sevent.h instance could reside in SEvt together with selector then hostside setup
can be common between the branches 

**/

void QEvent::init()
{
    LOG_IF(fatal, !sev) << "QEvent instanciated before SEvt instanciated : this is not going to fly " ; 

    assert(sev); 
    assert(evt); 
    assert(selector); 

    LOG(LEVEL) << " QEvent::init calling SEvt/setCompProvider " ; 
    sev->setCompProvider(this);  

    init_SEvt(); 
}

void QEvent::init_SEvt()
{
    if(SEvt_NPFold_VERBOSE)
    {
        LOG(info) << " QEvent__SEvt_NPFold_VERBOSE : setting SEvt:setFoldVerbose " ;
        sev->setFoldVerbose(true);
    }
}


std::string QEvent::desc() const
{
    std::stringstream ss ; 
    ss << evt->desc() << std::endl ;
    std::string s = ss.str();  
    return s ; 
}

std::string QEvent::desc_alloc() const
{
    salloc* alloc = QU::alloc ; 
    std::stringstream ss ; 
    ss << "[QEvent::desc_alloc " << std::endl ; 
    ss << ( alloc ? "salloc::desc" : "NO-salloc" ) << std::endl ;
    ss << ( alloc ? alloc->desc() : "" ) << std::endl ;
    ss << "]QEvent::desc_alloc " << std::endl ; 
    std::string s = ss.str();  
    return s ; 
}





/**
QEvent::setGenstep
-------------------

Canonically invoked from QSim::simulate and QSim::simtrace just prior to cx->launch 

Differences from _OLD version

1. no gatherGenstep, no intermediated NP array, instead 
   get genstep data direct from inside the vecs with no copying 

2. no SEvt::clear as that is too much dealing with 
   other lifecycle concerns that do not belong in
   a "setGenstep" method


**/

int QEvent::setGenstep()  // onto device
{
    LOG_IF(info, SEvt::LIFECYCLE) << "[" ; 


    NP* gs_ = sev->getGenstepArray();  
    int rc = setGenstepUpload_NP(gs_) ; 

    LOG_IF(info, SEvt::LIFECYCLE) << "]" ; 

    return rc ; 
}




/**
QEvent::setGenstepUpload_NP
------------------------------

**/
int QEvent::setGenstepUpload_NP(const NP* gs_ )
{
    return setGenstepUpload_NP(gs_, nullptr ); 
} 

int QEvent::setGenstepUpload_NP(const NP* gs_, const sslice* gss_ ) 
{
    gs = gs_ ; 
    gss = gss_ ? new sslice(*gss_) : nullptr ; 

    SGenstep::Check(gs); 

    LOG(LEVEL) 
        << " gs " << ( gs ? gs->sstr() : "-" )
        << SGenstep::Desc(gs, 10)
        ;

    int num_gs = gs ? gs->shape[0] : 0 ;

    int gs_start = gss ? gss->gs_start : 0 ; 
    int gs_stop  = gss ? gss->gs_stop  : num_gs ; 

    assert( gs_start >= 0 && gs_start <  num_gs ); 
    assert( gs_stop  >= 1 && gs_stop  <= num_gs ); 

    const char* data = gs ? gs->bytes() : nullptr ;
    const quad6* qq = (const quad6*)data ; 

    assert( gs_start < num_gs ); 
    assert( gs_stop <= num_gs );     

    int rc = setGenstepUpload(qq, gs_start, gs_stop ); 

    assert( gss->ph_count == evt->num_photon ); 
    // genstep consistency 

    int last_rng_state_idx = gss->ph_offset + gss->ph_count ; 
    bool in_range = last_rng_state_idx <= evt->max_curand ; 

    LOG_IF(error, !in_range)
        << " gss.desc " << gss->desc()
        << " gss->ph_offset " << gss->ph_offset
        << " gss->ph_count " << gss->ph_count
        << " gss->ph_offset + gss->ph_count " << last_rng_state_idx
        << " evt->max_curand " << evt->max_curand
        << " evt->num_curand " << evt->num_curand
        << " evt->max_slot " << evt->max_slot
        ; 

    assert( in_range );  
    // NB not max_slot as its the size of the RNG state array that matters 

    return rc ; 
}

int QEvent::getPhotonSlotOffset() const
{
    return gss ? gss->ph_offset : 0 ; 
}



/**
QEvent::setGenstepUpload
---------------------------

Switch to quad6* arg to allow direct from vector upload, 

Recall that even with input photon running, still have gensteps.  
If the number of gensteps is zero there are no photons and no launch. 


1. if not already allocated QEvent::device_alloc_genstep_and_seed
   using configured sevent::max_genstep sevent::max_photon values  

2. QU::copy_host_to_device the sevent::num_genstep 
   and setting pointer sevent::genstep 

3. QU::device_memset zeroing the seed buffer : this is needed 
   for each launch, doing at initialization only is not sufficient.
   **This is a documented limitation of sysrap/iexpand.h**

4. QEvent::count_genstep_photons_and_fill_seed_buffer

   * calculates the total number of seeds (and photons) on device 
     by adding the photons from each genstep and setting evt->num_seed

   * populates seed buffer using num photons per genstep from genstep buffer, 
     which is the way each photon thread refers back to its genstep

5. setNumSimtrace/setInputPhoton/setNumPhoton which may allocate records


**/

int QEvent::setGenstepUpload(const quad6* qq0, int num_gs )
{
    return setGenstepUpload(qq0, 0, num_gs ); 
} 
int QEvent::setGenstepUpload(const quad6* qq0, int gs_start, int gs_stop ) 
{
    const quad6* qq = qq0 + gs_start ; 


    LOG_IF(info, SEvt::LIFECYCLE) << "[" ; 
#ifndef PRODUCTION 
    sev->t_setGenstep_3 = sstamp::Now(); 
#endif

    int num_genstep = gs_stop - gs_start ; 
    bool zero_genstep = num_genstep == 0 ; 

    evt->num_genstep = num_genstep ; 
    bool not_allocated = evt->genstep == nullptr && evt->seed == nullptr ; 

    LOG_IF(info, LIFECYCLE) << " not_allocated " << ( not_allocated ? "YES" : "NO" ) ;  

    LOG(LEVEL) 
        << " gs_start " << gs_start
        << " gs_stop " << gs_stop
        << " evt.num_genstep " << evt->num_genstep 
        << " not_allocated " << ( not_allocated ? "YES" : "NO" ) 
        << " zero_genstep " << ( zero_genstep ? "YES" : "NO " )
        ; 

    if(not_allocated) 
    {
        device_alloc_genstep_and_seed() ; 
    }

 
    bool num_gs_allowed = evt->num_genstep <= evt->max_genstep ;
    LOG_IF(fatal, !num_gs_allowed) << " evt.num_genstep " << evt->num_genstep << " evt.max_genstep " << evt->max_genstep ; 
    assert( num_gs_allowed ); 

#ifndef PRODUCTION 
    sev->t_setGenstep_4 = sstamp::Now(); 
#endif

    if( qq != nullptr )
    QU::copy_host_to_device<quad6>( evt->genstep, (quad6*)qq, evt->num_genstep ); 

#ifndef PRODUCTION 
    sev->t_setGenstep_5 = sstamp::Now(); 
#endif

    QU::device_memset<int>(   evt->seed,    0, evt->max_photon );    // need evt->max_slot ? 

#ifndef PRODUCTION 
    sev->t_setGenstep_6 = sstamp::Now(); 
#endif

    if(num_genstep > 0)
    {
        //count_genstep_photons();   // sets evt->num_seed
        //fill_seed_buffer() ;       // populates seed buffer
        count_genstep_photons_and_fill_seed_buffer();   // combi-function doing what both the above do 
    }
    else
    {
        LOG(error) << " num_genstep ZERO : proceed anyhow eg for low level QSimTest tests" ;    
    }

#ifndef PRODUCTION 
    sev->t_setGenstep_7 = sstamp::Now(); 
#endif

    int gencode0 = SGenstep::GetGencode(qq, 0) ; // gencode of first genstep or OpticksGenstep_INVALID for qq nullptr  

    if(OpticksGenstep_::IsFrame(gencode0))   // OpticksGenstep_FRAME  (HMM: Obtuse, maybe change to SIMTRACE ?)
    {
        setNumSimtrace( evt->num_seed ); 
    }
    else if(OpticksGenstep_::IsInputPhoton(gencode0)) // OpticksGenstep_INPUT_PHOTON  (NOT: _TORCH)
    {
        setInputPhoton(); 
    }
    else
    {
        setNumPhoton( evt->num_seed );  // *HEAVY* : photon, rec, record may be allocated here depending on SEventConfig
    }
    upload_count += 1 ; 

#ifndef PRODUCTION 
    sev->t_setGenstep_8 = sstamp::Now(); 
#endif
    LOG_IF(info, SEvt::LIFECYCLE) << "]" ; 


    int rc = zero_genstep ? 1 : 0 ; 
    LOG_IF(error, rc != 0 ) << "No gensteps in SEvt::EGPU : ONLY OK WITH VERY LOW LEVEL TESTING eg QSimTest  " ; 

    return rc ; 
}






/**
QEvent::device_alloc_genstep_and_seed
-------------------------------------------

Allocates memory for genstep and seed, keeping device pointers within
the hostside sevent.h "evt->genstep" "evt->seed"

**/

void QEvent::device_alloc_genstep_and_seed()
{
    LOG_IF(info, LIFECYCLE) ; 
    LOG(LEVEL) << " device_alloc genstep and seed " ; 
    evt->genstep = QU::device_alloc<quad6>( evt->max_genstep, "QEvent::setGenstep/device_alloc_genstep_and_seed:quad6" ) ; 
    evt->seed    = QU::device_alloc<int>(   evt->max_photon , "QEvent::setGenstep/device_alloc_genstep_and_seed:int seed" )  ;
}



/**
QEvent::setInputPhoton
------------------------

This is a private method invoked only from QEvent::setGenstepUpload

1. SEvt::gatherInputPhoton narrows or copies the input 
   photons (which may be frame transformed) providing 
   a narrowed f4 array. 

   NB gatherInputPhoton always provides a fresh 
   unencumbered array that a subsequent SEvt::clear 
   cannot delete. So that means it just LEAKs, 
   but that currently not much of a problem 
   as input photons are used for debugging purposes 
   currently 

   TODO: WHEN DOING LEAK CHECKING TRY TO FIND THIS 
   LEAK AND AVOID IT BY DELETING THE ARRAY HERE 
   IMMEDIATELY AFTER UPLOAD 

   Input photons are awkward because they do not 
   follow the pattern of other arrays. They:

   * originate on the CPU (like gensteps)
   * have no dedicated device buffer for them (unlike gensteps)
   * get copied into the photons buffer instead of 
     being generated on device 
   * are not downloaded from device

   Effectively input photons are a cheat to avoid 
   on device generation that is convenuent for 
   debugging, and especially useful to provide
   common inputs for random aligned bi-simulation. 


2. QEvent::checkInputPhoton expectation asserts

3. QU::copy_host_to_device upload the input photon array 
   into the photon buffer

**/

void QEvent::setInputPhoton()
{
    LOG_IF(info, LIFECYCLE) ; 
    LOG(LEVEL); 
    input_photon = sev->gatherInputPhoton(); 
    checkInputPhoton(); 

    int numph = input_photon->shape[0] ; 
    setNumPhoton( numph ); 
    QU::copy_host_to_device<sphoton>( evt->photon, (sphoton*)input_photon->bytes(), numph ); 

    // HMM: there is a getter ... 
    //delete input_photon ; 
    //input_photon = nullptr ;  
}

void QEvent::checkInputPhoton() const 
{
    LOG_IF(fatal, input_photon == nullptr) 
        << " INCONSISTENT : OpticksGenstep_INPUT_PHOTON by no input photon array " 
        ; 

    assert(input_photon);  

    bool expected_shape = input_photon->has_shape( -1, 4, 4) ;
    bool expected_ebyte = input_photon->ebyte == 4 ;

    int numph = input_photon->shape[0] ; 
    bool expected_numph = evt->num_seed == numph ;

    LOG_IF(fatal, !expected_shape) << " !expected_shape " << input_photon->sstr() ; 
    LOG_IF(fatal, !expected_ebyte) << " !expected_ebyte " << input_photon->ebyte ; 
    LOG_IF(fatal, !expected_numph) << " !expected_numph " << numph << " evt.num_seed " << ( evt ? evt->num_seed : -1 )  ; 

    assert(expected_shape); 
    assert(expected_ebyte); 
    assert(expected_numph); 
}




/**
QEvent::setGenstep
-------------------

Was being used by  QEventTest::setGenstep_quad6

int QEvent::setGenstep(quad6* qgs, unsigned num_gs )  // TODO: what uses this ? eliminate ?
{
    NP* gs_ = NP::Make<float>( num_gs, 6, 4 ); 
    gs_->read2( (float*)qgs );   
    return setGenstepUpload( gs_ ); 
}
**/



// TODO: how to avoid duplication between QEvent and SEvt ?

bool QEvent::hasGenstep() const { return evt->genstep != nullptr ; }
bool QEvent::hasSeed() const {    return evt->seed != nullptr ; }
bool QEvent::hasPhoton() const {  return evt->photon != nullptr ; }
bool QEvent::hasRecord() const { return evt->record != nullptr ; }
bool QEvent::hasRec() const    { return evt->rec != nullptr ; }
bool QEvent::hasSeq() const    { return evt->seq != nullptr ; }
bool QEvent::hasPrd() const    { return evt->prd != nullptr ; }
bool QEvent::hasTag() const    { return evt->tag != nullptr ; }
bool QEvent::hasFlat() const   { return evt->flat != nullptr ; }
bool QEvent::hasHit() const    { return evt->hit != nullptr ; }
bool QEvent::hasSimtrace() const  { return evt->simtrace != nullptr ; }




/**
QEvent::count_genstep_photons
------------------------------

thrust::reduce using strided iterator summing over GPU side gensteps 

**/

extern "C" unsigned QEvent_count_genstep_photons(sevent* evt) ; 
unsigned QEvent::count_genstep_photons()
{
   return QEvent_count_genstep_photons( evt );  
}

/**
QEvent::fill_seed_buffer
---------------------------

Populates seed buffer using the number of photons from each genstep 

The photon seed buffer is a device buffer containing integer indices referencing 
into the genstep buffer. The seeds provide the association between the photon 
and the genstep required to generate it.

**/

extern "C" void QEvent_fill_seed_buffer(sevent* evt ); 
void QEvent::fill_seed_buffer()
{
    LOG_IF(info, LIFECYCLE) ; 
    QEvent_fill_seed_buffer( evt ); 
}

extern "C" void QEvent_count_genstep_photons_and_fill_seed_buffer(sevent* evt ); 
void QEvent::count_genstep_photons_and_fill_seed_buffer()
{
    LOG_IF(info, LIFECYCLE) ; 
    QEvent_count_genstep_photons_and_fill_seed_buffer( evt ); 
}




NP* QEvent::getGenstep() const 
{
    NP* _gs = const_cast<NP*>(gs) ; // const_cast so can use QEvent::gatherComponent_
    LOG(LEVEL) << " _gs " << ( _gs ? _gs->sstr() : "-" ) ; 
    return _gs ; 
}
NP* QEvent::getInputPhoton() const 
{
    return input_photon ; 
}







/**
QEvent::gatherPhoton(NP* p) :  mutating API
-------------------------------------------

* QU::copy_device_to_host using (sevent)evt->photon/num_photon

  * sevent.h needs changing for each sub-launch 


**/

void QEvent::gatherPhoton(NP* p) const 
{
    bool expected_shape =  p->has_shape(evt->num_photon, 4, 4) ; 
    LOG(expected_shape ? LEVEL : fatal) << "[ evt.num_photon " << evt->num_photon << " p.sstr " << p->sstr() << " evt.photon " << evt->photon ; 
    assert(expected_shape ); 
    int rc = QU::copy_device_to_host<sphoton>( (sphoton*)p->bytes(), evt->photon, evt->num_photon ); 

    LOG_IF(fatal, rc != 0) 
         << " QU::copy_device_to_host photon FAILED "
         << " evt->photon " << ( evt->photon ? "Y" : "N" ) 
         << " evt->num_photon " <<  evt->num_photon
         ;

    if(rc != 0) std::raise(SIGINT) ; 

    LOG(LEVEL) << "] evt.num_photon " << evt->num_photon  ; 
}

NP* QEvent::gatherPhoton() const 
{
    //NP* p = NP::Make<float>( evt->num_photon, 4, 4); 
    NP* p = sev->makePhoton(); 
    gatherPhoton(p); 
    return p ; 
}


#ifndef PRODUCTION

NP* QEvent::gatherSeed() const 
{
    bool has_seed = hasSeed() ; 
    LOG_IF(fatal, !has_seed) << " gatherSeed called when there is no such array, use SEventConfig::SetCompMask to avoid " ; 
    if(!has_seed) return nullptr ;  
    NP* s = NP::Make<int>( evt->num_seed );   // TODO: use SEvt::makeSeed
    QU::copy_device_to_host<int>( (int*)s->bytes(), evt->seed, evt->num_seed ); 
    return s ; 
}

NP* QEvent::gatherDomain() const { return sev ? sev->gatherDomain() : nullptr ; }


/**
QEvent::gatherGenstepFromDevice
---------------------------------

Gensteps originate on host and are uploaded to device, so downloading
them from device is not usually done. It is for debugging only. 

**/

NP* QEvent::gatherGenstepFromDevice() const 
{
    NP* a = NP::Make<float>( evt->num_genstep, 6, 4 );  
    QU::copy_device_to_host<quad6>( (quad6*)a->bytes(), evt->genstep, evt->num_genstep ); 
    return a ; 
}


void QEvent::gatherSimtrace(NP* t) const 
{
    LOG(LEVEL) << "[ evt.num_simtrace " << evt->num_simtrace << " t.sstr " << t->sstr() << " evt.simtrace " << evt->simtrace ; 
    assert( t->has_shape(evt->num_simtrace, 4, 4) ); 
    QU::copy_device_to_host<quad4>( (quad4*)t->bytes(), evt->simtrace, evt->num_simtrace ); 
    LOG(LEVEL) << "] evt.num_simtrace " << evt->num_simtrace  ; 
}
NP* QEvent::gatherSimtrace() const 
{
    bool has_simtrace = hasSimtrace(); 
    LOG_IF(LEVEL, !has_simtrace) << " getSimtrace called when there is no such array, use SEventConfig::SetCompMask to avoid " ;
    if(!has_simtrace) return nullptr ;  
    NP* t = NP::Make<float>( evt->num_simtrace, 4, 4);   // TODO: use SEvt::makeSimtrace ?
    gatherSimtrace(t); 
    return t ; 
}

void QEvent::gatherSeq(NP* seq) const 
{
    bool has_seq = hasSeq(); 
    if(!has_seq) return ; 
    LOG(LEVEL) << "[ evt.num_seq " << evt->num_seq << " seq.sstr " << seq->sstr() << " evt.seq " << evt->seq ; 
    assert( seq->has_shape(evt->num_seq, 2) ); 
    QU::copy_device_to_host<sseq>( (sseq*)seq->bytes(), evt->seq, evt->num_seq ); 
    LOG(LEVEL) << "] evt.num_seq " << evt->num_seq  ; 
}
NP* QEvent::gatherSeq() const 
{
    bool has_seq = hasSeq(); 
    LOG_IF(LEVEL, !has_seq) << " gatherSeq called when there is no such array, use SEventConfig::SetCompMask to avoid " ; 
    if(!has_seq) return nullptr ;
  
    NP* seq = sev->makeSeq(); 

    gatherSeq(seq); 
    return seq ; 
}



NP* QEvent::gatherPrd() const 
{
    bool has_prd = hasPrd(); 
    LOG_IF(LEVEL, !has_prd) << " gatherPrd called when there is no such array, use SEventConfig::SetCompMask to avoid " ; 
    if(!has_prd) return nullptr ;
  
    NP* prd = sev->makePrd(); 
    LOG(LEVEL) << " evt.num_prd " << evt->num_prd ; 
    QU::copy_device_to_host<quad2>( (quad2*)prd->bytes(), evt->prd, evt->num_prd ); 
    return prd ; 
}

NP* QEvent::gatherTag() const 
{
    bool has_tag = hasTag() ; 
    LOG_IF(LEVEL, !has_tag) << " gatherTag called when there is no such array, use SEventConfig::SetCompMask to avoid " ; 
    if(!has_tag) return nullptr ;
  
    NP* tag = sev->makeTag(); 
    LOG(LEVEL) << " evt.num_tag " << evt->num_tag << " tag.desc " << tag->desc() ; 
    QU::copy_device_to_host<stag>( (stag*)tag->bytes(), evt->tag, evt->num_tag ); 
    return tag ; 
}

NP* QEvent::gatherFlat() const 
{
    bool has_flat = hasFlat(); 
    LOG_IF(LEVEL, !has_flat) << " gatherFlat called when there is no such array, use SEventConfig::SetCompMask to avoid " ; 
    if(!has_flat) return nullptr ;
  
    NP* flat = sev->makeFlat(); 
    LOG(LEVEL) << " evt.num_flat " << evt->num_flat << " flat.desc " << flat->desc() ; 
    QU::copy_device_to_host<sflat>( (sflat*)flat->bytes(), evt->flat, evt->num_flat ); 
    return flat ; 
}


NP* QEvent::gatherRecord() const 
{
    bool has_record = hasRecord() ; 
    LOG_IF(LEVEL, !has_record) << " gatherRecord called when there is no such array, use SEventConfig::SetCompMask to avoid " ; 
    if(!has_record) return nullptr ;  

    NP* r = sev->makeRecord(); 

    LOG(LEVEL) << " evt.num_record " << evt->num_record ; 
    QU::copy_device_to_host<sphoton>( (sphoton*)r->bytes(), evt->record, evt->num_record ); 
    return r ; 
}

NP* QEvent::gatherRec() const 
{
    NP* r = nullptr ; 
    bool has_rec = hasRec(); 
    LOG_IF(LEVEL, !has_rec ) << " gatherRec called when there is no such array, use SEventConfig::SetCompMask to avoid " ; 
    if(!has_rec) return nullptr ;  

    r = sev->makeRec(); 

    LOG(LEVEL) 
        << " evt.num_photon " << evt->num_photon 
        << " evt.max_rec " << evt->max_rec 
        << " evt.num_rec " << evt->num_rec  
        << " evt.num_photon*evt.max_rec " << evt->num_photon*evt->max_rec  
        ;

    assert( evt->num_photon*evt->max_rec == evt->num_rec );  

    QU::copy_device_to_host<srec>( (srec*)r->bytes(), evt->rec, evt->num_rec ); 
    return r ; 
}
#endif

/**
QEvent::getNumHit  TODO:rejig 
-----------------------------------

HMM: applies selector to the GPU photon array, thats surprising 
for a "get" method... TODO: maybe rearrange to do that once only 
at the gatherHit stage and subsequently just get the count from 
SEvt::fold 

**/


unsigned QEvent::getNumHit() const  
{
    assert( evt->photon ); 
    assert( evt->num_photon ); 
    LOG_IF(info, LIFECYCLE) ; 

    evt->num_hit = SU::count_if_sphoton( evt->photon, evt->num_photon, *selector );    

    LOG(LEVEL) << " evt.photon " << evt->photon << " evt.num_photon " << evt->num_photon << " evt.num_hit " << evt->num_hit ;  
    return evt->num_hit ; 
}

/**
QEvent::gatherHit
------------------

1. on device count *evt.num_hit* passing the photon *selector* 
2. allocate *evt.hit* GPU buffer
3. copy_if from *evt.photon* to *evt.hit* using the photon *selector*
4. host allocate the NP hits array
5. copy hits from device to the host NP hits array 
6. free *evt.hit* on device
7. return NP hits array to caller, who becomes owner of the array 

Note that the device hits array is allocated and freed for each launch.  
This is due to the expectation that the number of hits will vary greatly from launch to launch 
unlike the number of photons which is expected to be rather similar for most launches other than 
remainder last launches. 

The alternative to this dynamic "busy" handling of hits would be to reuse a fixed hits buffer
sized to max_photons : that however seems unpalatable due it always doubling up GPU memory for 
photons and hits.  

hitmask metadata was formerly placed on the hit array, 
subsequently moved to domain_meta as domain should 
always be present, unlike hits. 

**/

NP* QEvent::gatherHit() const 
{
    // hasHit at this juncture is misleadingly always false, 
    // because the hits array is derived by *gatherHit_* which  selects from the photons 

    bool has_photon = hasPhoton(); 

    LOG_IF(LEVEL, !has_photon) << " gatherHit called when there is no photon array " ; 
    if(!has_photon) return nullptr ; 

    assert( evt->photon ); 

    LOG_IF(fatal, evt->num_photon == 0 ) << " evt->num_photon ZERO " ;  
    assert( evt->num_photon ); 

    evt->num_hit = SU::count_if_sphoton( evt->photon, evt->num_photon, *selector );    

    LOG(LEVEL) 
         << " evt.photon " << evt->photon 
         << " evt.num_photon " << evt->num_photon 
         << " evt.num_hit " << evt->num_hit
         << " selector.hitmask " << selector->hitmask
         << " SEventConfig::HitMask " << SEventConfig::HitMask()
         << " SEventConfig::HitMaskLabel " << SEventConfig::HitMaskLabel()
         ;  

    NP* hit = evt->num_hit > 0 ? gatherHit_() : nullptr ; 

    return hit ; 
}

NP* QEvent::gatherHit_() const 
{
    LOG_IF(info, LIFECYCLE) ; 
    evt->hit = QU::device_alloc<sphoton>( evt->num_hit, "QEvent::gatherHit_:sphoton" ); 

    SU::copy_if_device_to_device_presized_sphoton( evt->hit, evt->photon, evt->num_photon,  *selector );

    NP* hit = NP::Make<float>( evt->num_hit, 4, 4 );   // TODO: use SEvt::makeHit ?

    QU::copy_device_to_host<sphoton>( (sphoton*)hit->bytes(), evt->hit, evt->num_hit );

    QU::device_free<sphoton>( evt->hit ); 

    evt->hit = nullptr ; 
    LOG(LEVEL) << " hit.sstr " << hit->sstr() ; 

    return hit ; 
}


/**
QEvent::getMeta
-----------------

SCompProvider method 

**/

std::string QEvent::getMeta() const 
{     
    return sev->meta ; 
}

const char* QEvent::getTypeName() const
{
    return TYPENAME ; 
}

/**
QEvent::gatherComponent
------------------------

Invoked for example by SEvt::gather_components via the SCompProvider protocol

**/

NP* QEvent::gatherComponent(unsigned cmp) const 
{
    LOG(LEVEL) << "[ cmp " << cmp ; 
    unsigned gather_mask = SEventConfig::GatherComp(); 
    bool proceed = (gather_mask & cmp) != 0 ; 
    NP* a = proceed ? gatherComponent_(cmp) : nullptr ;
    LOG(LEVEL) << "[ cmp " << cmp << " proceed " << proceed << " a " <<  a ; 
    return a ; 
}

/**
QEvent::gatherComponent_
-------------------------

Gather downloads from device, get accesses from host 

**/

NP* QEvent::gatherComponent_(unsigned cmp) const 
{
    NP* a = nullptr ; 
    switch(cmp)
    {   
        //case SCOMP_GENSTEP:   a = getGenstep()     ; break ;   
        case SCOMP_GENSTEP:   a = gatherGenstepFromDevice() ; break ;   
        case SCOMP_INPHOTON:  a = getInputPhoton() ; break ;   

        case SCOMP_PHOTON:    a = gatherPhoton()   ; break ;   
        case SCOMP_HIT:       a = gatherHit()      ; break ;   
#ifndef PRODUCTION
        case SCOMP_DOMAIN:    a = gatherDomain()      ; break ;   
        case SCOMP_RECORD:    a = gatherRecord()   ; break ;   
        case SCOMP_REC:       a = gatherRec()      ; break ;   
        case SCOMP_SEQ:       a = gatherSeq()      ; break ;   
        case SCOMP_PRD:       a = gatherPrd()      ; break ;   
        case SCOMP_SEED:      a = gatherSeed()     ; break ;   
        case SCOMP_SIMTRACE:  a = gatherSimtrace() ; break ;   
        case SCOMP_TAG:       a = gatherTag()      ; break ;   
        case SCOMP_FLAT:      a = gatherFlat()     ; break ;   
#endif
    }   
    return a ; 
}



/**
QEvent::setNumPhoton
---------------------

At the first call when evt.photon is nullptr allocation on device is done. 

Canonically invoked internally from QEvent::setGenstep but may be invoked 
directly from "friendly" photon only tests without use of gensteps.  

1. Sets evt->num_photon which asserts that is within allowed *evt->max_photon* 
2. allocates buffers for all configured arrays (how heavy depends on configured array sizes)
3. calls *uploadEvt* (lightweight, just counts and pointers)

This assumes that the number of photons for subsequent launches does not increase 
when collecting records : that is ok as running with records is regarded as debugging. 

**/

void QEvent::setNumPhoton(unsigned num_photon )
{
    LOG_IF(info, LIFECYCLE) << " num_photon " << num_photon ; 
    LOG(LEVEL); 

    sev->setNumPhoton(num_photon); 
    if( evt->photon == nullptr ) device_alloc_photon();  
    uploadEvt(); 
}


void QEvent::setNumSimtrace(unsigned num_simtrace)
{
    sev->setNumSimtrace(num_simtrace); 
    if( evt->simtrace == nullptr ) device_alloc_simtrace(); 
    uploadEvt(); 
}






/**
QEvent::device_alloc_photon
----------------------------

Buffers are allocated on device and the device pointers are collected 
into hostside sevent.h "evt" 

**/

void QEvent::device_alloc_photon()
{
    LOG_IF(info, LIFECYCLE) ; 
    SetAllocMeta( QU::alloc, evt );   // do this first as memory errors likely to happen in following lines

    LOG(LEVEL) 
        << " evt.max_photon " << evt->max_photon 
        << " evt.num_photon " << evt->num_photon 
#ifndef PRODUCTION
        << " evt.num_record " << evt->num_record 
        << " evt.num_rec    " << evt->num_rec 
        << " evt.num_seq    " << evt->num_seq
        << " evt.num_prd    " << evt->num_prd
        << " evt.num_tag    " << evt->num_tag
        << " evt.num_flat   " << evt->num_flat
#endif
        ;

    evt->photon  = evt->max_photon > 0 ? QU::device_alloc_zero<sphoton>( evt->max_photon, "QEvent::device_alloc_photon/max_photon*sizeof(sphoton)" ) : nullptr ; 

#ifndef PRODUCTION
    evt->record  = evt->max_record > 0 ? QU::device_alloc_zero<sphoton>( evt->max_photon * evt->max_record, "max_photon*max_record*sizeof(sphoton)" ) : nullptr ; 
    evt->rec     = evt->max_rec    > 0 ? QU::device_alloc_zero<srec>(    evt->max_photon * evt->max_rec   , "max_photon*max_rec*sizeof(srec)"    ) : nullptr ; 
    evt->prd     = evt->max_prd    > 0 ? QU::device_alloc_zero<quad2>(   evt->max_photon * evt->max_prd   , "max_photon*max_prd*sizeof(quad2)"    ) : nullptr ; 
    evt->seq     = evt->max_seq   == 1 ? QU::device_alloc_zero<sseq>(    evt->max_photon                  , "max_photon*sizeof(sseq)"    ) : nullptr ; 
    evt->tag     = evt->max_tag   == 1 ? QU::device_alloc_zero<stag>(    evt->max_photon                  , "max_photon*sizeof(stag)"    ) : nullptr ; 
    evt->flat    = evt->max_flat  == 1 ? QU::device_alloc_zero<sflat>(   evt->max_photon                  , "max_photon*sizeof(sflat)"   ) : nullptr ; 
#endif

    LOG(LEVEL) << desc() ; 
    LOG(LEVEL) << desc_alloc() ; 
}





void QEvent::SetAllocMeta(salloc* alloc, const sevent* evt)  // static 
{
    if(!alloc) return ; 
    if(!evt) return ; 
    evt->get_meta(alloc->meta); 
}

 
void QEvent::device_alloc_simtrace()
{
    LOG_IF(info, LIFECYCLE) ; 
    evt->simtrace = QU::device_alloc<quad4>( evt->max_simtrace, "QEvent::device_alloc_simtrace/max_simtrace" ) ; 
    LOG(LEVEL) 
        << " evt.num_simtrace " << evt->num_simtrace 
        << " evt.max_simtrace " << evt->max_simtrace
        ;
}


/**
QEvent::uploadEvt 
--------------------

Uploads lightweight sevent.h instance with counters and pointers for the array. 

Copies host side sevent.h *evt* instance (with updated num_genstep and num_photon) to device side  *d_evt*.  
Note that the evt->genstep and evt->photon pointers are not updated, so the same buffers are reused for each launch. 

**/

void QEvent::uploadEvt()
{
    LOG_IF(info, LIFECYCLE) ; 
    LOG(LEVEL) << std::endl << evt->desc() ; 
    QU::copy_host_to_device<sevent>(d_evt, evt, 1 );  
}

unsigned QEvent::getNumPhoton() const
{
    return evt->num_photon ; 
}
unsigned QEvent::getNumSimtrace() const
{
    return evt->num_simtrace ; 
}



extern "C" void QEvent_checkEvt(dim3 numBlocks, dim3 threadsPerBlock, sevent* evt, unsigned width, unsigned height ) ; 

void QEvent::checkEvt() 
{ 
    unsigned width = getNumPhoton() ; 
    unsigned height = 1 ; 
    LOG(info) << " width " << width << " height " << height ; 

    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 
    QU::ConfigureLaunch( numBlocks, threadsPerBlock, width, height ); 
 
    assert( d_evt ); 
    QEvent_checkEvt(numBlocks, threadsPerBlock, d_evt, width, height );   
}


