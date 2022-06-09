
#include "scuda.h"
#include "squad.h"
#include "sphoton.h"
#include "srec.h"
#include "sseq.h"
#include "sevent.h"


#include "PLOG.hh"
#include "SSys.hh"
#include "NP.hh"
#include "NPFold.h"
#include "SPath.hh"
#include "SGeo.hh"
#include "SEvt.hh"
#include "SEvent.hh"
#include "SEventConfig.hh"
#include "OpticksGenstep.h"
#include "OpticksPhoton.h"
#include "OpticksPhoton.hh"
#include "SComp.h"

const plog::Severity SEvt::LEVEL = PLOG::EnvLevel("SEvt", "DEBUG"); 
const int SEvt::GIDX = SSys::getenvint("GIDX",-1) ;


SEvt* SEvt::INSTANCE = nullptr ; 

SEvt::SEvt()
    :
    selector(new sphoton_selector(SEventConfig::HitMask())),
    evt(new sevent),
    fold(new NPFold)
{ 
    init(); 
}

/**
SEvt::init
-----------

Only configures limits, no allocation yet. 
Device side allocation happens in QEvent::setGenstep QEvent::setNumPhoton

Initially SEvt is set as its own SCompProvider, 
allowing U4RecorderTest/SEvt::save to gather the component 
arrays provided from SEvt.

For device running the SCompProvider  is overridden to 
become QEvent allowing SEvt::save to persist the 
components gatherered from device buffers. 

**/

void SEvt::init()
{
    INSTANCE = this ; 
    evt->init(); 

    setCompProvider(this); // overridden for device running from QEvent::init 
    LOG(fatal) << evt->desc() ;
}

void SEvt::setCompProvider(const SCompProvider* provider_)
{
    provider = provider_ ; 
}


NP* SEvt::getDomain() const
{
    quad4 dom[2] ;
    evt->get_domain(dom[0]);
    evt->get_config(dom[1]);
    NP* domain = NP::Make<float>( 2, 4, 4 );
    domain->read2<float>( (float*)&dom[0] );
    // actually it makes more sense to place metadata on domain than hits 
    // as domain will always be available
    domain->set_meta<unsigned>("hitmask", selector->hitmask );
    domain->set_meta<std::string>("creator", "SEvt::getDomain" );
    return domain ;
}

SEvt* SEvt::Get(){ return INSTANCE ; }
void SEvt::Check()
{
    if(INSTANCE == nullptr) std::cout << "FATAL: must instanciate SEvt before using most SEvt methods" << std::endl ; 
    assert(INSTANCE); 
}


sgs SEvt::AddGenstep(const quad6& q){ Check(); return INSTANCE->addGenstep(q);  }
sgs SEvt::AddGenstep(const NP* a){    Check(); return INSTANCE->addGenstep(a); }
void SEvt::AddCarrierGenstep(){ AddGenstep(SEvent::MakeCarrierGensteps()); }
void SEvt::AddTorchGenstep(){   AddGenstep(SEvent::MakeTorchGensteps());   }

void SEvt::Clear(){ Check() ; INSTANCE->clear();  }
void SEvt::Save(){  Check() ; INSTANCE->save(); }
void SEvt::Save(const char* dir){                  Check() ; INSTANCE->save(dir); }
void SEvt::Save(const char* dir, const char* rel){ Check() ; INSTANCE->save(dir, rel ); }

int SEvt::GetNumPhoton(){ return INSTANCE ? INSTANCE->getNumPhoton() : -1 ; }
NP* SEvt::GetGenstep() {  return INSTANCE ? INSTANCE->getGenstep() : nullptr ; }

void SEvt::clear()
{
    genstep.clear();
    gs.clear();
    pho0.clear(); 
    pho.clear(); 
    slot.clear(); 
    photon.clear(); 
    record.clear(); 
    rec.clear(); 
    seq.clear(); 
}

unsigned SEvt::getNumGenstep() const 
{
    assert( genstep.size() == gs.size() ); 
    return genstep.size() ; 
}

unsigned SEvt::getNumPhoton() const 
{
    unsigned tot = 0 ; 
    for(unsigned i=0 ; i < genstep.size() ; i++) tot += genstep[i].numphoton() ; 
    return tot ; 
}

/**
SEvt::addGenstep
------------------

The sgs summary struct of the last genstep is returned. 

**/

sgs SEvt::addGenstep(const NP* a)
{
    int num_gs = a ? a->shape[0] : -1 ; 
    assert( num_gs > 0 ); 
    quad6* qq = (quad6*)a->bytes(); 
    sgs s = {} ; 
    for(int i=0 ; i < num_gs ; i++) s = addGenstep(qq[i]) ; 
    return s ; 
}

bool SEvt::RECORDING = true ;  // TODO: needs to be normally false

/**
SEvt::addGenstep
------------------

The GIDX envvar is used whilst debugging to restrict to collecting 
a single genstep chosen by its index.  This is implemented by 
always collecting all genstep labels, but only collecting 
actual gensteps for the enabled index. 

**/

sgs SEvt::addGenstep(const quad6& q_)
{
    int gidx = int(gs.size())  ;  // 0-based genstep label index
    bool enabled = GIDX == -1 || GIDX == gidx ; 

    quad6& q = const_cast<quad6&>(q_);   
    if(!enabled) q.set_numphoton(0);   
    // simplify handling of disabled gensteps by simply setting numphoton to zero for them

    sgs s = {} ;                  // genstep summary struct 
    s.index = genstep.size() ;    // 0-based genstep index since last clear  
    s.photons = q.numphoton() ;   // numphoton in this genstep 
    s.offset = getNumPhoton() ;   // sum numphotons from all previously collected gensteps (since last reset)
    s.gentype = q.gentype() ; 

    gs.push_back(s) ; 
    genstep.push_back(q) ; 

    if(enabled) LOG(info) << " s.desc " << s.desc() << " gidx " << gidx << " enabled " << enabled  ; 

    int tot_photon = s.offset+s.photons ; 
    if( tot_photon != evt->num_photon )
    {
        setNumPhoton(tot_photon); 
        resize();  
    }

    // TODO: work out what needs to NOT be done (eg resize) 
    //       for on device running as opposed to U4Recorder running 

    return s ; 
}

/**
SEvt::resize
-------------

This is the CPU side equivalent of device side QEvent::setNumPhoton

TODO: use SEvt::setNumPhoton from QEvent::setNumPhoton to avoid the duplicity 

**/

void SEvt::setNumPhoton(unsigned numphoton)
{
    LOG(info) << " numphoton " << numphoton ;  

    evt->num_photon = numphoton ; 
    evt->num_seq    = evt->max_seq > 0 ? evt->num_photon : 0 ;
    evt->num_record = evt->max_record * evt->num_photon ;
    evt->num_rec    = evt->max_rec    * evt->num_photon ;
}

void SEvt::resize()
{
    if(evt->num_photon > 0) pho.resize(  evt->num_photon );  
    if(evt->num_photon > 0) slot.resize( evt->num_photon ); 

    if(evt->num_photon > 0) photon.resize(evt->num_photon);
    if(evt->num_record > 0) record.resize(evt->num_record); 
    if(evt->num_rec    > 0) rec.resize(evt->num_rec); 
    if(evt->num_seq    > 0) seq.resize(evt->num_seq); 

    if(evt->num_photon > 0) evt->photon = photon.data() ; 
    if(evt->num_record > 0) evt->record = record.data() ; 
    if(evt->num_rec    > 0) evt->rec    = rec.data() ; 
    if(evt->num_seq    > 0) evt->seq    = seq.data() ; 
}



/**
SEvt::get_gs
--------------

Lookup sgs genstep label corresponding to spho photon label 

**/

const sgs& SEvt::get_gs(const spho& label) const 
{
    assert( label.gs < int(gs.size()) ); 
    const sgs& _gs =  gs[label.gs] ; 
    return _gs ; 
}

unsigned SEvt::get_genflag(const spho& label) const 
{
    const sgs& _gs = get_gs(label);  
    int gentype = _gs.gentype ;
    unsigned genflag = OpticksGenstep_::GenstepToPhotonFlag(gentype); 
    assert( genflag == CERENKOV || genflag == SCINTILLATION || genflag == TORCH ); 
    return genflag ; 
}


/**
SEvt::beginPhoton
------------------


**/
void SEvt::beginPhoton(const spho& label)
{
    LOG(info) ; 
    LOG(info) << label.desc() ; 

    unsigned idx = label.id ; 

    bool in_range = idx < pho.size() ; 
    if(!in_range) LOG(error) 
        << " not in_range " 
        << " idx " << idx 
        << " pho.size  " << pho.size() 
        << " label " << label.desc() 
        ;  
    assert(in_range);  

    unsigned genflag = get_genflag(label);  

    pho0.push_back(label);    // push_back asis for debugging
    pho[idx] = label ;        // slot in the photon label  
    slot[idx] = 0 ;           // slot/bounce incremented only at tail of SEvt::pointPhoton

    current_pho = label ; 

    current_photon.zero() ; 
    current_rec.zero() ; 
    current_seq.zero() ; 

    current_photon.set_idx(idx); 
    current_photon.set_flag(genflag); 

    assert( current_photon.flagmask_count() == 1 ); // should only be a single bit in the flagmask at this juncture 
}

/**
SEvt::rjoinPhoton
----------------------

Called from U4Recorder::PreUserTrackingAction_Optical for G4Track with 
spho label indicating a reemission generation greater than zero.

Note that this will mostly be called for photons that originate from 
scintillation gensteps BUT it will also happen for Cerenkov (and Torch) genstep 
generated photons within a scintillator due to reemission. 

TODO: check that positions match up across the rejoin 

**/
void SEvt::rjoinPhoton(const spho& label)
{
    LOG(info); 
    LOG(info) << label.desc() ; 

    unsigned idx = label.id ; 
    assert( idx < pho.size() );  

    // check labels of parent and child are as expected
    const spho& parent_label = pho[idx]; 
    assert( label.isSameLineage( parent_label) ); 
    assert( label.gn == parent_label.gn + 1 ); 

    const sgs& _gs = get_gs(label);  
    bool expected_gentype = OpticksGenstep_::IsExpected(_gs.gentype); 
    assert(expected_gentype);  
    // within a scintillator the photons from any genstep type may undergo reemission  

    const sphoton& parent_photon = photon[idx] ; 
    unsigned parent_idx = parent_photon.idx() ; 
    assert( parent_idx == idx ); 

    // replace label and current_photon
    pho[idx] = label ;   
    current_pho = label ; 

    int& bounce = slot[idx] ; assert( bounce > 0 );   
    int prior = bounce - 1 ; 

    if( evt->photon )
    {
        // HMM: could directly change photon[idx] via ref ? 
        // But are here taking a copy to current_photon
        // and relying on copyback at SEvt::endPhoton

        current_photon = photon[idx] ; 
        rjoinPhotonCheck(current_photon); 

        current_photon.flagmask &= ~BULK_ABSORB  ; // scrub BULK_ABSORB from flagmask
        current_photon.set_flag(BULK_REEMIT) ;     // gets OR-ed into flagmask 
    }

    if( evt->seq )
    {
        current_seq = seq[idx] ; 
        unsigned seq_flag = current_seq.get_flag(prior);
        bool seq_flag_AB = seq_flag == BULK_ABSORB ;
        if(seq_flag_AB == false) std::cout << " NOT seq_flag_AB, rather " << OpticksPhoton::Abbrev(seq_flag) << std::endl ;  
        //assert( seq_flag_AB ); 
        current_seq.set_flag(prior, BULK_REEMIT);  
    }

    if( evt->record )
    {
        sphoton& rjoin_record = evt->record[evt->max_record*idx+prior]  ; 
        std::string rjoin_record_d12 = rjoin_record.digest(12) ; 
        std::string current_photon_d12 = current_photon.digest(12) ; 
        bool d12_match = strcmp( rjoin_record_d12.c_str(), current_photon_d12.c_str() ) == 0 ;  

        std::cout 
            << " rjoin_record_d12   " << rjoin_record_d12  << std::endl
            << " current_photon_d12 " << current_photon_d12 << std::endl
            << " d12_match " << ( d12_match ? "YES" : "NO" ) << std::endl
            ;
        assert( d12_match ); 

        std::cout 
            << " rjoin_record " 
            << std::endl 
            << rjoin_record.desc()
            << std::endl 
            ;
 
        unsigned rjoin_flag = rjoin_record.flag() ; 
        LOG(info) << " rjoin.flag "  << OpticksPhoton::Flag(rjoin_flag)  ; 

        bool rjoin_flag_AB = rjoin_flag == BULK_ABSORB  ; 
        bool rjoin_record_flagmask_AB = rjoin_record.flagmask & BULK_ABSORB ; 

        if(!rjoin_flag_AB) std::cout << " NOT rjoin_flag_AB " << std::endl ; 
        if(!rjoin_record_flagmask_AB) std::cout << " NOT rjoin_record_flagmask_AB " << std::endl ; 

        //assert( rjoin_flag_AB ); 
        //assert( rjoin_record_flagmask_AB ); 

        rjoin_record.flagmask &= ~BULK_ABSORB ; // scrub BULK_ABSORB from flagmask  
        rjoin_record.set_flag(BULK_REEMIT) ; 

        std::cout 
            << " current_photon "
            << std::endl 
            << current_photon.desc()
            << std::endl 
            ;

    } 
    // TODO: rec  (compressed record)
}

/**
SEvt::rjoinPhotonCheck
------------------------

Would expect all rejoin to have BULK_ABSORB ? 

What about perhaps reemission immediately after reemission  ?

**/

void SEvt::rjoinPhotonCheck(const sphoton& ph ) const 
{
    bool flag_AB     = ph.flag() == BULK_ABSORB ;  
    bool flagmask_AB = ph.flagmask & BULK_ABSORB  ; 
    if(!(flag_AB && flagmask_AB))
    {
        std::cout 
            << "rjoinPhotonCheck : does not have BULK_ABSORB flag ?" 
            << " ph.idx " << ph.idx() 
            << " flag_AB " << ( flag_AB ? "YES" : "NO" )
            << " flagmask_AB " << ( flagmask_AB ? "YES" : "NO" )
            << std::endl
            << ph.desc()
            << std::endl
            ;
    }
    //assert( flag_AB ); 
    //assert( flagmask_AB  );   
}



/**
SEvt::pointPhoton
------------------

Invoked from U4Recorder::UserSteppingAction_Optical to cause the 
current photon to be recorded into record vector. 

The pointPhoton and finalPhoton methods need to do the hostside equivalent of 
what CSGOptiX/CSGOptiX7.cu:simulate does device side,
so have setup the environment to match::

As the hostside vectors keep getting resized at each genstep, the 
evt buffer points are updated at every resize to follow them around
as they grow and are reallocated.

TODO: truncation : bounce < max_bounce 

**/

void SEvt::pointPhoton(const spho& label)
{
    assert( label.isSameLineage(current_pho) ); 
    unsigned idx = label.id ; 
    int& bounce = slot[idx] ; 

    const sphoton& p = current_photon ; 
    srec& rec        = current_rec ; 
    sseq& seq        = current_seq ; 

    if( evt->record && bounce < evt->max_record ) evt->record[evt->max_record*idx+bounce] = p ;   
    if( evt->rec    && bounce < evt->max_rec    ) evt->add_rec(rec, idx, bounce, p );  
    if( evt->seq    && bounce < evt->max_seq    ) seq.add_nibble(bounce, p.flag(), p.boundary() );

    LOG(info) << label.desc() << " seqhis: " << OpticksPhoton::FlagSequence( seq.seqhis ) ; 

    bounce += 1 ; 
}

void SEvt::finalPhoton(const spho& label)
{
    LOG(info) << label.desc() ; 
    assert( label.isSameLineage(current_pho) ); 
    unsigned idx = label.id ; 

    const sphoton& p = current_photon ; 
    sseq& seq        = current_seq ; 

    if(evt->photon) evt->photon[idx] = p ; 
    if(evt->seq)    evt->seq[idx] = seq ; 
}

void SEvt::checkPhoton(const spho& label) const 
{
    assert( label.isSameLineage(current_pho) ); 
}



////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
///////// below methods handle gathering arrays and persisting, not array content //////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////


NP* SEvt::getPho0() const { return NP::Make<int>( (int*)pho0.data(), int(pho0.size()), 4 ); }
NP* SEvt::getPho() const {  return NP::Make<int>( (int*)pho.data(), int(pho.size()), 4 ); }
NP* SEvt::getGS() const {   return NP::Make<int>( (int*)gs.data(),  int(gs.size()), 4 );  }


NP* SEvt::getPhoton() const 
{ 
    if( evt->photon == nullptr ) return nullptr ; 
    NP* p = makePhoton(); 
    p->read2( (float*)evt->photon ); 
    return p ; 
} 
NP* SEvt::getRecord() const 
{ 
    if( evt->record == nullptr ) return nullptr ; 
    NP* r = makeRecord(); 
    r->read2( (float*)evt->record ); 
    return r ; 
} 
NP* SEvt::getRec() const 
{ 
    if( evt->rec == nullptr ) return nullptr ; 
    NP* r = makeRec(); 
    r->read2( (short*)evt->rec ); 
    return r ; 
} 
NP* SEvt::getSeq() const 
{ 
    if( evt->seq == nullptr ) return nullptr ; 
    NP* s = makeSeq(); 
    s->read2( (unsigned long long*)evt->seq ); 
    return s ; 
} 



NP* SEvt::makePhoton() const 
{
    return NP::Make<float>( evt->num_photon, 4, 4 ); 
}
NP* SEvt::makeRecord() const 
{ 
    NP* r = NP::Make<float>( evt->num_photon, evt->max_record, 4, 4 ); 
    r->set_meta<std::string>("rpos", "4,GL_FLOAT,GL_FALSE,64,0,false" );  // eg used by examples/UseGeometryShader
    return r ; 
}
NP* SEvt::makeRec() const 
{
    NP* r = NP::Make<short>( evt->num_photon, evt->max_rec, 2, 4);   // stride:  sizeof(short)*2*4 = 2*2*4 = 16   
    r->set_meta<std::string>("rpos", "4,GL_SHORT,GL_TRUE,16,0,false" );  // eg used by examples/UseGeometryShader
    return r ; 
}
NP* SEvt::makeSeq() const 
{
    return NP::Make<unsigned long long>( evt->num_seq, 2); 
}

// SCompProvider methods

std::string SEvt::getMeta() const 
{
    return meta ; 
}

NP* SEvt::getComponent(unsigned comp) const 
{
    unsigned mask = SEventConfig::CompMask(); 
    return mask & comp ? getComponent_(comp) : nullptr ; 
}
NP* SEvt::getComponent_(unsigned comp) const 
{
    NP* a = nullptr ; 
    switch(comp)
    {   
        case SCOMP_GENSTEP:   a = getGenstep()  ; break ;   
        case SCOMP_PHOTON:    a = getPhoton()   ; break ;   
        case SCOMP_RECORD:    a = getRecord()   ; break ;   
        case SCOMP_REC:       a = getRec()      ; break ;   
        case SCOMP_SEQ:       a = getSeq()      ; break ;   
        //case SCOMP_SEED:      a = getSeed()     ; break ;   
        //case SCOMP_HIT:       a = getHit()      ; break ;   
        //case SCOMP_SIMTRACE:  a = getSimtrace() ; break ;   
        case SCOMP_DOMAIN:    a = getDomain()   ; break ;   
    }   
    return a ; 
}

/**
SEvt::saveLabels
--------------

**/

void SEvt::saveLabels(const char* dir_) const 
{
    const char* dir = SPath::Resolve(dir_, DIRPATH );  
    LOG(info) << dir ; 

    NP* a0 = getPho0();  
    LOG(info) << " a0 " << ( a0 ? a0->sstr() : "-" ) ; 
    if(a0) a0->save(dir, "pho0.npy"); 

    NP* a = getPho();  
    LOG(info) << " a " << ( a ? a->sstr() : "-" ) ; 
    if(a) a->save(dir, "pho.npy"); 

    NP* g = getGS(); 
    LOG(info) << " g " << ( g ? g->sstr() : "-" ) ; 
    if(g) g->save(dir, "gs.npy"); 
}

/**
SEvt::getGenstep
-----------------

The returned array takes a full copy of the genstep quad6 vector
with all gensteps collected since the last SEvt::clear. 
The array is thus independent from quad6 vector, and hence is untouched
by SEvt::clear 

**/

NP* SEvt::getGenstep() const { return NP::Make<float>( (float*)genstep.data(), int(genstep.size()), 6, 4 ) ; }

void SEvt::saveGenstep(const char* dir) const  // HMM: NOT THE STANDARD SAVE 
{
    NP* a = getGenstep(); 
    if(a == nullptr) return ; 
    LOG(LEVEL) << a->sstr() << " dir " << dir ; 
    a->save(dir, "gs.npy"); 
}

std::string SEvt::descGS() const 
{
    std::stringstream ss ; 
    for(unsigned i=0 ; i < getNumGenstep() ; i++) ss << gs[i].desc() << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
}

std::string SEvt::desc() const 
{
    std::stringstream ss ; 
    ss << evt->desc() ; 
    std::string s = ss.str(); 
    return s ; 
}

/**
SEvt::gather_components
--------------------------

Collects the components configured by SEventConfig::CompMask
into NPFold by for example downloading from the QEvent provider. 

**/

void SEvt::gather_components() 
{
    unsigned mask = SEventConfig::CompMask();
    std::vector<unsigned> comps ; 
    SComp::CompListAll(comps );
    for(unsigned i=0 ; i < comps.size() ; i++)
    {
        unsigned comp = comps[i] ;   
        if((comp & mask) == 0) continue ; 
        NP* a = provider->getComponent(comp); 
        if(a == nullptr) continue ;  
        const char* k = SComp::Name(comp);    
        fold->add(k, a); 
    }
    fold->meta = provider->getMeta();  
    // persisted metadata will now be in NPFold_meta.txt (previously fdmeta.txt)
}

std::string SEvt::descFold() const 
{
    return fold->desc(); 
}



/**
SEvt::save
--------------

This was formerly implemented up in qudarap/QEvent but it makes no 
sense for CPU only tests that need to save events to reach up to qudarap 
to control persisting. 

The component arrays are gathered by SEvt::gather_components
into the NPFold and then saved. Which components to gather and save 
are configured via SEventConfig::SetCompMask using the SComp enumeration. 

The arrays are gathered from the SCompProvider object, which 
may be QEvent for on device running or SEvt itself for U4Recorder 
Geant4 tests. 

SEvt::save persists NP arrays into the default directory 
or the directory argument provided.
**/

const char* SEvt::FALLBACK_DIR = "$TMP" ; 
const char* SEvt::DefaultDir()
{
    const char* dir_ = SGeo::LastUploadCFBase_OutDir(); 
    const char* dir = dir_ ? dir_ : FALLBACK_DIR  ; 
    return dir ; 
}

void SEvt::save() 
{
    const char* dir = DefaultDir(); 
    LOG(info) << "DefaultDir " << dir ; 
    save(dir); 
}
void SEvt::save(const char* base, const char* reldir ) 
{
    const char* dir = SPath::Resolve(base, reldir, DIRPATH); 
    save(dir); 
}
void SEvt::save(const char* dir_) 
{
    const char* dir = SPath::Resolve(dir_, DIRPATH); 
    LOG(info) << " dir " << dir ; 

    gather_components(); 

    LOG(info) << descComponent() ; 
    LOG(info) << descFold() ; 

    fold->save(dir); 
}


std::string SEvt::descComponent() const 
{
    const NP* genstep  = fold->get(SComp::Name(SCOMP_GENSTEP)) ; 
    const NP* seed     = fold->get(SComp::Name(SCOMP_SEED)) ;  
    const NP* photon   = fold->get(SComp::Name(SCOMP_PHOTON)) ; 
    const NP* hit      = fold->get(SComp::Name(SCOMP_HIT)) ; 
    const NP* record   = fold->get(SComp::Name(SCOMP_RECORD)) ; 
    const NP* rec      = fold->get(SComp::Name(SCOMP_REC)) ;  
    const NP* seq      = fold->get(SComp::Name(SCOMP_SEQ)) ; 
    const NP* domain   = fold->get(SComp::Name(SCOMP_DOMAIN)) ; 
    const NP* simtrace = fold->get(SComp::Name(SCOMP_SIMTRACE)) ; 

    std::stringstream ss ; 
    ss << "SEvt::descComponent" 
       << std::endl 
       << std::setw(20) << " SEventConfig::CompMaskLabel " << SEventConfig::CompMaskLabel() << std::endl  
       << std::setw(20) << "hit" << " " 
       << std::setw(20) << ( hit ? hit->sstr() : "-" ) 
       << " "
       << std::endl
       << std::setw(20) << "seed" << " " 
       << std::setw(20) << ( seed ? seed->sstr() : "-" ) 
       << " "
       << std::endl
       << std::setw(20) << "genstep" << " " 
       << std::setw(20) << ( genstep ? genstep->sstr() : "-" ) 
       << " "
       << std::setw(30) << "SEventConfig::MaxGenstep" 
       << std::setw(20) << SEventConfig::MaxGenstep()
       << std::endl

       << std::setw(20) << "photon" << " " 
       << std::setw(20) << ( photon ? photon->sstr() : "-" ) 
       << " "
       << std::setw(30) << "SEventConfig::MaxPhoton"
       << std::setw(20) << SEventConfig::MaxPhoton()
       << std::endl
       << std::setw(20) << "record" << " " 
       << std::setw(20) << ( record ? record->sstr() : "-" ) 
       << " " 
       << std::setw(30) << "SEventConfig::MaxRecord"
       << std::setw(20) << SEventConfig::MaxRecord()
       << std::endl
       << std::setw(20) << "rec" << " " 
       << std::setw(20) << ( rec ? rec->sstr() : "-" ) 
       << " "
       << std::setw(30) << "SEventConfig::MaxRec"
       << std::setw(20) << SEventConfig::MaxRec()
       << std::endl
       << std::setw(20) << "seq" << " " 
       << std::setw(20) << ( seq ? seq->sstr() : "-" ) 
       << " " 
       << std::setw(30) << "SEventConfig::MaxSeq"
       << std::setw(20) << SEventConfig::MaxSeq()
       << std::endl
       << std::setw(20) << "domain" << " " 
       << std::setw(20) << ( domain ? domain->sstr() : "-" ) 
       << " "
       << std::endl
       << std::setw(20) << "simtrace" << " " 
       << std::setw(20) << ( simtrace ? simtrace->sstr() : "-" ) 
       << " "
       << std::endl
       ;
    std::string s = ss.str(); 
    return s ; 
}


