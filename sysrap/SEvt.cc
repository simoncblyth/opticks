
#include "scuda.h"
#include "squad.h"
#include "sphoton.h"
#include "srec.h"
#include "sseq.h"
#include "sevent.h"

#include "PLOG.hh"
#include "NP.hh"
#include "NPFold.h"
#include "SPath.hh"
#include "SGeo.hh"
#include "SEvt.hh"
#include "SEvent.hh"
#include "SEventConfig.hh"
#include "OpticksGenstep.h"
#include "OpticksPhoton.h"
#include "SComp.h"

const plog::Severity SEvt::LEVEL = PLOG::EnvLevel("SEvt", "DEBUG"); 

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

sgs SEvt::addGenstep(const quad6& q)
{
    unsigned offset = getNumPhoton() ; // number of photons in event before this genstep  (actually since last reset) 
    unsigned q_numphoton = q.numphoton() ; 

    sgs s = {} ;                // genstep summary struct 
    s.index = genstep.size() ;  // 0-based genstep index in event (actually since last reset)  
    s.photons = q_numphoton ;   // numphoton in the genstep 
    s.offset = offset ;         // event global photon offset 
    s.gentype = q.gentype() ; 

    LOG(info) << " s.desc " << s.desc() ; 

    // gs labels and gensteps in order of collection
    gs.push_back(s) ; 
    genstep.push_back(q) ; 

    if(RECORDING) 
    {
        // numphotons from all gensteps in event so far plus this one just added
        setNumPhoton(offset + q_numphoton); 
        resize();  
    }

    return s ; 
}

/**
SEvt::resize
-------------

This is the CPU side equivalent of device side QEvent::setNumPhoton

**/

void SEvt::setNumPhoton(unsigned numphoton)
{
    // TODO: use SEvt::setNumPhoton from QEvent::setNumPhoton to avoid the duplicity 
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

const sgs& SEvt::get_gs(const spho& sp)
{
    assert( sp.gs < int(gs.size()) ); 
    const sgs& _gs =  gs[sp.gs] ; 
    return _gs ; 
}

/**
SEvt::beginPhoton
------------------


**/
void SEvt::beginPhoton(const spho& sp)
{
    unsigned idx = sp.id ; 
    assert( idx < pho.size() );  

    pho0.push_back(sp);   // push_back asis : just for initial dev, TODO: remove 

    pho[idx] = sp ;        // slot in the label  
    slot[idx] = 0 ; 
    current_pho = sp ; 

    const sgs& _gs = get_gs(sp);  
    int gentype = _gs.gentype ;
    unsigned genflag = OpticksGenstep_::GenstepToPhotonFlag(gentype); 
    assert( genflag == CERENKOV || genflag == SCINTILLATION || genflag == TORCH ); 

    LOG(info) << " _gs " << _gs.desc() ; 

    current_photon.zero() ; 
    current_photon.set_idx(idx); 
    current_photon.set_flag(genflag); 
    assert( current_photon.flagmask_count() == 1 ); 
}

/**
SEvt::continuePhoton
----------------------

Called from U4Recorder::PreUserTrackingAction_Optical for G4Track with 
pho label indicating a reemission generation greater than zero.

Note that this will mostly be called for photons that originate from 
scintillation gensteps BUT it will also happen for Cerenkov genstep 
generated photons within a scintillator due to reemission of the Cerenkov photons. 

**/
void SEvt::continuePhoton(const spho& sp)
{
    unsigned idx = sp.id ; 
    assert( idx < pho.size() );  

    // check labels of parent and child are as expected
    const spho& parent_pho = pho[idx]; 
    assert( sp.isSameLineage( parent_pho) ); 
    assert( sp.gn == parent_pho.gn + 1 ); 

    const sgs& _gs = get_gs(sp);  
    bool sc = OpticksGenstep_::IsScintillation(_gs.gentype); 
    bool ck = OpticksGenstep_::IsCerenkov(_gs.gentype); 
    bool sc_xor_ck = sc ^ ck ; 
    LOG(info) << " sc " << sc << " ck " << ck << " sc_xor_ck " << sc_xor_ck ;
    assert(sc_xor_ck); 
 
    const sphoton& parent_photon = photon[idx] ; 
    unsigned parent_idx = parent_photon.idx() ; 
    assert( parent_idx == idx ); 

    // replace label and current_photon
    pho[idx] = sp ;   
    current_pho = sp ; 

    // HMM: could directly change photon[idx] via ref ? 
    // But are here taking a copy to current_photon, and relying on copyback at SEvt::endPhoton
    current_photon = photon[idx] ; 

    //assert( current_photon.flagmask & BULK_ABSORB  );   // all continuedPhoton should have BULK_ABSORB in flagmask, but not yet 

    int& _slot = slot[idx] ; 
    assert( _slot > 0 );   
    // _slot -= 1 ;  
    // back up the slot by one : HMM maybe not needed as pointPhoton will just be called for post steppoint

    current_photon.flagmask &= ~BULK_ABSORB  ; // scrub BULK_ABSORB from flagmask
    current_photon.set_flag(BULK_REEMIT) ;     // gets OR-ed into flagmask 
}


/**
SEvt::pointPhoton
------------------

Invoked from U4Recorder::UserSteppingAction_Optical to cause the 
current photon to be recorded into record vector. 

TODO: truncation : bounce < max_bounce 

**/

void SEvt::pointPhoton(const spho& sp)
{
    assert( sp.isSameLineage(current_pho) ); 
    unsigned idx = sp.id ; 
    int& bounce = slot[idx] ; 

    const sphoton& p = current_photon ; 
    srec& rec = current_rec ; 
    sseq& seq = current_seq ; 

    if( evt->record && bounce < evt->max_record ) evt->record[evt->max_record*idx+bounce] = p ;   
    if( evt->rec    && bounce < evt->max_rec    ) evt->add_rec(rec, idx, bounce, p );  
    if( evt->seq    && bounce < evt->max_seq    ) seq.add_step(bounce, p.flag(), p.boundary() );

    bounce += 1 ; 
}

/**
These methods need to do the hostside equivalent of CSGOptiX/CSGOptiX7.cu:simulate device side,
so have setup the environment to match::

    if( evt->record && bounce < evt->max_record ) evt->record[evt->max_record*idx+bounce] = p ;   
    if( evt->rec    && bounce < evt->max_rec    ) evt->add_rec(rec, idx, bounce, p );  
    if( evt->seq    && bounce < evt->max_seq    ) seq.add_step(bounce, p.flag(), p.boundary() );

    evt->photon[idx] = p ; 
    if(evt->seq) evt->seq[idx] = seq ;

As the hostside vectors keep getting resized at each genstep, the 
evt buffer points are updated at every resize to follow them around
as they grow and are realloced. 
**/

void SEvt::endPhoton(const spho& sp)
{
    assert( sp.isSameLineage(current_pho) ); 
    unsigned idx = sp.id ; 

    const sphoton& p = current_photon ; 
    sseq& seq = current_seq ; 
    
    if(evt->photon) evt->photon[idx] = p ; 
    if(evt->seq)    evt->seq[idx] = seq ; 
}

/**
SEvt::checkPhoton
-------------------

Called from  U4Recorder::UserSteppingAction

**/

void SEvt::checkPhoton(const spho& sp) const 
{
    assert( sp.isSameLineage(current_pho) ); 
}


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

   // NP* p = getPhoton(); 
   // LOG(info) << " p " << ( p ? p->sstr() : "-" ) ; 
   // if(p) p->save(dir, "p.npy"); 

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


The component arrays are downloaded from the device by SEvt::gather_components
that are added to the NPFold and then saved. 

Which components to gather and save is configured via SEventConfig::SetCompMask
using the SComp enumeration. 

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


