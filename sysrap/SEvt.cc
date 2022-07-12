
#include <limits>

#include "scuda.h"
#include "squad.h"
#include "sphoton.h"
#include "srec.h"
#include "sseq.h"
#include "sstate.h"
#include "stag.h"
#include "sevent.h"
#include "sctx.h"
#include "sdebug.h"
#include "stran.h"


#include "PLOG.hh"
#include "SSys.hh"
#include "SStr.hh"
#include "NP.hh"
#include "NPFold.h"
#include "SPath.hh"
#include "SGeo.hh"
#include "SEvt.hh"
#include "SEvent.hh"
#include "SEventConfig.hh"
#include "SFrameGenstep.hh"
#include "OpticksGenstep.h"
#include "OpticksPhoton.h"
#include "OpticksPhoton.hh"
#include "SComp.h"
#include "SCF.h"

const plog::Severity SEvt::LEVEL = PLOG::EnvLevel("SEvt", "DEBUG"); 
const int SEvt::GIDX = SSys::getenvint("GIDX",-1) ;
const int SEvt::PIDX = SSys::getenvint("PIDX",-1) ;
const int SEvt::MISSING_INDEX = std::numeric_limits<int>::max() ; 
const SCF* SEvt::CF = SCF::Create() ;

SEvt* SEvt::INSTANCE = nullptr ; 

SEvt::SEvt()
    :
    index(MISSING_INDEX),
    reldir(nullptr),
    selector(new sphoton_selector(SEventConfig::HitMask())),
    evt(new sevent),
    dbg(new sdebug),
    input_photon(nullptr),
    input_photon_transformed(nullptr),
    random(nullptr),
    provider(this),   // overridden with SEvt::setCompProvider for device running from QEvent::init 
    fold(new NPFold),
    hostside_running_resize_done(false)
{ 
    init(); 
}

const char* SEvt::getSaveDir() const 
{
   return fold->savedir ;  
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
    dbg->zero(); 
    LOG(LEVEL) << evt->desc() ;

    initInputPhoton(); 
}


const char* SEvt::INPUT_PHOTON_DIR = SSys::getenvvar("SEvt_INPUT_PHOTON_DIR", "$HOME/.opticks/InputPhotons") ; 

/**
SEvt::LoadInputPhoton
----------------------

Resolving the input string to a path is done in one of two ways:

1. if the string starts with a letter A-Za-z eg "inphoton.npy" or "RandomSpherical10.npy" 
   it is assumed to be the name of a .npy file within the default SEvt_INPUT_PHOTON_DIR 
   of $HOME/.opticks/InputPhotons. 

   Create such files with ana/input_photons.py  

2. if the string does not start with a letter eg /path/to/some/dir/file.npy or $TOKEN/path/to/file.npy 
   it is passed unchanged to  SPath::Resolve

**/

NP* SEvt::LoadInputPhoton() // static 
{
    const char* ip = SEventConfig::InputPhoton(); 
    return ip ? LoadInputPhoton(ip) : nullptr ; 
}

NP* SEvt::LoadInputPhoton(const char* ip)
{
    assert(strlen(ip) > 0 && SStr::EndsWith(ip, ".npy") ); 
    const char* path = SStr::StartsWithLetterAZaz(ip) ?  SPath::Resolve(INPUT_PHOTON_DIR, ip, NOOP) : SPath::Resolve( ip, NOOP ) ; 

    NP* a = NP::Load(path); 
    if( a == nullptr ) LOG(fatal) << " FAILED to load input photon from path " << path << " SEventConfig::InputPhoton " << ip ; 

    assert( a ) ; 
    assert( a->has_shape(-1,4,4) ); 
    assert( a->shape[0] > 0 );  

    LOG(LEVEL) 
        << " SEventConfig::InputPhoton " << ip
        << " path " << path 
        << " a.sstr " << a->sstr()
        ;

    return a ; 
}



/**
SEvt::initInputPhoton
-----------------------

This is invoked by SEvt::init on instanciating the SEvt instance  
The default "SEventConfig::InputPhoton()" is nullptr meaning no input photons.
This can be changed by setting an envvar in the script that runs the executable, eg::

   export OPTICKS_INPUT_PHOTON=CubeCorners.npy
   export OPTICKS_INPUT_PHOTON=$HOME/reldir/path/to/inphoton.npy
 
Or within the code of the executable, typically in the main prior to SEvt instanciation, 
using eg::

   SEventConfig::SetInputPhoton("CubeCorners.npy")
   SEventConfig::SetInputPhoton("$HOME/reldir/path/to/inphoton.npy")

When non-null it is resolved into a path and the array loaded at SEvt instanciation
by SEvt::LoadInputPhoton

**/

void SEvt::initInputPhoton()
{
    NP* ip = LoadInputPhoton() ;
    setInputPhoton(ip); 
}

void SEvt::setInputPhoton(NP* p)
{
    if(p == nullptr) return ; 
    input_photon = p ; 
    assert( input_photon->has_shape(-1,4,4) ); 
    int numphoton = input_photon->shape[0] ; 
    assert( numphoton > 0 ); 
}
 
NP* SEvt::getInputPhoton_() const { return input_photon ; }
NP* SEvt::getInputPhoton() const {  return input_photon_transformed ? input_photon_transformed : input_photon  ; }
bool SEvt::hasInputPhoton() const { return input_photon != nullptr ; }



/**
SEvt::setFrame
------------------

As it is necessary to have the geometry to provide the frame this 
is now split from eg initInputPhotons.  

For simtrace and input photon running with or without a transform 
it is necessary to call this prior to calling QSim::simulate.

**/


void SEvt::setFrame(const sframe& fr )
{
    frame = fr ; 

    if(SEventConfig::IsRGModeSimtrace())
    {   
        addGenstep( SFrameGenstep::MakeCenterExtentGensteps(frame) );  
    }   
    else if(SEventConfig::IsRGModeSimulate() && hasInputPhoton())
    {   
        assert( genstep.size() == 0 ) ; // cannot mix input photon running with other genstep running  

        addGenstep(MakeInputPhotonGenstep(input_photon, frame)); 

        input_photon_transformed = frame.transform_photon_m2w( input_photon ); 
    }   
}


quad6 SEvt::MakeInputPhotonGenstep(const NP* input_photon, const sframe& fr )
{
    quad6 ipgs ; 
    ipgs.zero(); 
    ipgs.set_gentype( OpticksGenstep_INPUT_PHOTON ); 
    ipgs.set_numphoton(  input_photon->shape[0]  ); 
    fr.m2w.write(ipgs); // copy fr.m2w into ipgs.q2,q3,q4,q5 
    return ipgs ; 
}



void SEvt::setCompProvider(const SCompProvider* provider_)
{
    provider = provider_ ; 
    LOG(LEVEL) << descProvider() ; 
}

bool SEvt::isSelfProvider() const {   return provider == this ; }

std::string SEvt::descProvider() const 
{
    bool is_self_provider = isSelfProvider() ; 
    std::stringstream ss ; 
    ss << "SEvt::descProvider provider: " << provider << " that address is: " << ( is_self_provider ? "SELF" : "another object" ) ; 
    std::string s = ss.str(); 
    return s ; 
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


void SEvt::AddTag(unsigned stack, float u ){  INSTANCE->addTag(stack,u);  } 
int  SEvt::GetTagSlot(){ return INSTANCE->getTagSlot() ; }
sgs SEvt::AddGenstep(const quad6& q){ Check(); return INSTANCE->addGenstep(q);  }
sgs SEvt::AddGenstep(const NP* a){    Check(); return INSTANCE->addGenstep(a); }
void SEvt::AddCarrierGenstep(){ AddGenstep(SEvent::MakeCarrierGensteps()); }
void SEvt::AddTorchGenstep(){   AddGenstep(SEvent::MakeTorchGensteps());   }

void SEvt::Clear(){ Check() ; INSTANCE->clear();  }
void SEvt::Save(){  Check() ; INSTANCE->save(); }
void SEvt::Save(const char* dir){                  Check() ; INSTANCE->save(dir); }
void SEvt::Save(const char* dir, const char* rel){ Check() ; INSTANCE->save(dir, rel ); }

void SEvt::SetIndex(int index){ assert(INSTANCE) ; INSTANCE->setIndex(index) ; }
void SEvt::UnsetIndex(){        assert(INSTANCE) ; INSTANCE->unsetIndex() ;  }
int SEvt::GetIndex(){           return INSTANCE ? INSTANCE->getIndex()  :  0 ; }



// SetReldir can be used with the default SEvt::save() changing the last directory element before the index if present
void        SEvt::SetReldir(const char* reldir){ assert(INSTANCE) ; INSTANCE->setReldir(reldir) ; }
const char* SEvt::GetReldir(){  return INSTANCE ? INSTANCE->getReldir() : nullptr ; }

int SEvt::GetNumPhoton(){ return INSTANCE ? INSTANCE->getNumPhoton() : -1 ; }

NP* SEvt::GetGenstep() {      return INSTANCE ? INSTANCE->getGenstep() : nullptr ; }
NP* SEvt::GetInputPhoton() {  return INSTANCE ? INSTANCE->getInputPhoton() : nullptr ; }
bool SEvt::HasInputPhoton(){  return INSTANCE ? INSTANCE->hasInputPhoton() : false ; }



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

void SEvt::setIndex(int index_){ index = index_ ; }
void SEvt::unsetIndex(){         index = MISSING_INDEX ; }
int SEvt::getIndex() const { return index ; }

void SEvt::setReldir(const char* reldir_){ reldir = reldir_ ? strdup(reldir_) : nullptr ; }
const char* SEvt::getReldir() const { return reldir ; }


/**
SEvt::getNumPhotonFromGenstep
----------------------------------


**/
unsigned SEvt::getNumPhoton() const 
{
   return getNumPhotonFromGenstep() ; 
}

unsigned SEvt::getNumPhotonFromGenstep() const 
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
    dbg->addGenstep++ ; 


    unsigned gentype = q_.gentype(); 
    bool input_photon_with_normal_genstep = input_photon && OpticksGenstep_::IsInputPhoton(gentype) == false  ; 
    if(input_photon_with_normal_genstep)
    {
        LOG(fatal)
            << "input_photon_with_normal_genstep " << input_photon_with_normal_genstep
            << " MIXING input photons with other gensteps is not allowed "
            << " for example avoid defining OPTICKS_INPUT_PHOTON when doing simtrace"
            ; 
    }
    assert( input_photon_with_normal_genstep  == false ); 


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

    int tot_photon = s.offset+s.photons ; 

    if(enabled) LOG(LEVEL) << " s.desc " << s.desc() << " gidx " << gidx << " enabled " << enabled << " tot_photon " << tot_photon ; 

    if( tot_photon != evt->num_photon )
    {
        setNumPhoton(tot_photon); 

    }
    return s ; 
}

/**
SEvt::resize
-------------

This is the CPU side equivalent of device side QEvent::setNumPhoton

TODO: use SEvt::setNumPhoton from QEvent::setNumPhoton to avoid the duplicity 

**/

void SEvt::setNumPhoton(unsigned num_photon)
{
    bool num_photon_allowed = int(num_photon) <= evt->max_photon ; 
    if(!num_photon_allowed) LOG(fatal) << " num_photon " << num_photon << " evt.max_photon " << evt->max_photon ;
    assert( num_photon_allowed );
    LOG(LEVEL) 
        << " num_photon " << num_photon 
        ;  

    evt->num_photon = num_photon ; 
    evt->num_seq    = evt->max_seq  > 0 ? evt->num_photon : 0 ;
    evt->num_tag    = evt->max_tag  > 0 ? evt->num_photon : 0 ;
    evt->num_flat   = evt->max_flat > 0 ? evt->num_photon : 0 ;

    evt->num_record = evt->max_record * evt->num_photon ;
    evt->num_rec    = evt->max_rec    * evt->num_photon ;
    evt->num_prd    = evt->max_prd    * evt->num_photon ;

    LOG(LEVEL)
        << " evt->num_photon " << evt->num_photon
        << " evt->num_tag " << evt->num_tag
        << " evt->num_flat " << evt->num_flat
        ;

    hostside_running_resize_done = false ; 
}

void SEvt::setNumSimtrace(unsigned num_simtrace)
{
    bool num_simtrace_allowed = int(num_simtrace) <= evt->max_simtrace ;
    if(!num_simtrace_allowed) LOG(fatal) << " num_simtrace " << num_simtrace << " evt.max_simtrace " << evt->max_simtrace ;
    assert( num_simtrace_allowed );
    LOG(LEVEL) << " num_simtrace " << num_simtrace ;  

    evt->num_simtrace = num_simtrace ;
}


/**
SEvt::hostside_running_resize
-------------------------------

NB evt->photon, evt->record, ...  pointers are used to hold device side 
addresses for QEvent running and CPU side addresses for SEvt based running.

This makes it necessary to defer resizing, to give QEvent 
time to be instanciated and declare itself as the comp provider 

**/

void SEvt::hostside_running_resize()
{
    bool is_self_provider = isSelfProvider() ; 
    LOG(LEVEL) 
        << " is_self_provider " << is_self_provider 
        << " hostside_running_resize_done " << hostside_running_resize_done
        ;

    assert( hostside_running_resize_done == false ); 
    assert( is_self_provider ); 
    hostside_running_resize_done = true ; 


    if(evt->num_photon > 0) pho.resize(  evt->num_photon );  
    if(evt->num_photon > 0) slot.resize( evt->num_photon ); 

     // HMM: what about tag_slot 

    if(evt->num_photon > 0) 
    { 
        photon.resize(evt->num_photon);
        evt->photon = photon.data() ; 
    }
    if(evt->num_record > 0) 
    {
        record.resize(evt->num_record); 
        evt->record = record.data() ; 
    }
    if(evt->num_rec > 0) 
    {
        rec.resize(evt->num_rec); 
        evt->rec = rec.data() ; 
    }
    if(evt->num_seq > 0) 
    {
        seq.resize(evt->num_seq); 
        evt->seq = seq.data() ; 
    }
    if(evt->num_prd > 0) 
    {
        prd.resize(evt->num_prd); 
        evt->prd = prd.data() ; 
    }
    if(evt->num_tag > 0) 
    {
        tag.resize(evt->num_tag); 
        evt->tag = tag.data() ; 
    }
    if(evt->num_flat > 0) 
    {
        flat.resize(evt->num_flat); 
        evt->flat = flat.data() ; 
    }

    LOG(LEVEL) 
        << " is_self_provider " << is_self_provider 
        << std::endl 
        << evt->desc() 
        ; 
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
    if(!hostside_running_resize_done) hostside_running_resize(); 

    dbg->beginPhoton++ ; 
    LOG(LEVEL) ; 
    LOG(LEVEL) << label.desc() ; 

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
    current_prd.zero() ;   

    sctx& ctx = current_ctx ; 
    ctx.zero(); 

    //LOG(info) << " idx " << idx ; 

    ctx.idx = idx ;  
    ctx.evt = evt ; 
    ctx.prd = &current_prd ;   // current_prd is populated by InstrumentedG4OpBoundaryProcess::PostStepDoIt 

    ctx.p.set_idx(idx); 
    ctx.p.set_flag(genflag); 

    assert( ctx.p.flagmask_count() == 1 ); // should only be a single bit in the flagmask at this juncture 
}

unsigned SEvt::getCurrentPhotonIdx() const { return current_pho.id ; }


/**
SEvt::rjoinPhoton
----------------------

Called from U4Recorder::PreUserTrackingAction_Optical for G4Track with 
spho label indicating a reemission generation greater than zero.

Note that this will mostly be called for photons that originate from 
scintillation gensteps BUT it will also happen for Cerenkov (and Torch) genstep 
generated photons within a scintillator due to reemission. 

HMM: could directly change photon[idx] via ref ? 
But are here taking a copy to current_photon
and relying on copyback at SEvt::endPhoton

**/
void SEvt::rjoinPhoton(const spho& label)
{
    dbg->rjoinPhoton++ ; 
    LOG(LEVEL); 
    LOG(LEVEL) << label.desc() ; 

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
        //current_photon = photon[idx] ; 
        current_ctx.p = photon[idx] ; 
        sphoton& current_photon = current_ctx.p ; 

        rjoinPhotonCheck(current_photon); 
        current_photon.flagmask &= ~BULK_ABSORB  ; // scrub BULK_ABSORB from flagmask
        current_photon.set_flag(BULK_REEMIT) ;     // gets OR-ed into flagmask 
    }

    // at truncation point and beyond cannot compare or do rejoin fixup
    if( evt->seq && prior < evt->max_seq )
    {
        //current_seq = seq[idx] ; 
        current_ctx.seq = seq[idx] ; 
        sseq& current_seq = current_ctx.seq ; 

        unsigned seq_flag = current_seq.get_flag(prior);
        rjoinSeqCheck(seq_flag); 
        current_seq.set_flag(prior, BULK_REEMIT);  
    }

    // at truncation point and beyond cannot compare or do rejoin fixup
    if( evt->record && prior < evt->max_record )  
    {
        sphoton& rjoin_record = evt->record[evt->max_record*idx+prior]  ; 
        rjoinRecordCheck(rjoin_record, current_ctx.p); 
        rjoin_record.flagmask &= ~BULK_ABSORB ; // scrub BULK_ABSORB from flagmask  
        rjoin_record.set_flag(BULK_REEMIT) ; 
    } 
    // TODO: rec  (compressed record)
}


void SEvt::rjoinRecordCheck(const sphoton& rj, const sphoton& ph  ) const
{
    assert( rj.idx() == ph.idx() );
    unsigned idx = rj.idx();  
    bool d12match = sphoton::digest_match( rj, ph, 12 ); 
    if(!d12match) dbg->d12match_fail++ ; 
    if(!d12match) ComparePhotonDump(rj, ph) ; 
    if(!d12match) std::cout 
        << " idx " << idx
        << " slot[idx] " << slot[idx]
        << " evt.max_record " << evt->max_record 
        << " d12match " << ( d12match ? "YES" : "NO" ) << std::endl
        ;
    assert( d12match ); 
}

void SEvt::ComparePhotonDump(const sphoton& a, const sphoton& b )  // static
{
    unsigned a_flag = a.flag() ; 
    unsigned b_flag = a.flag() ; 
    std::cout 
        << " a.flag "  << OpticksPhoton::Flag(a_flag)  
        << std::endl 
        << a.desc()
        << std::endl 
        << " b.flag "  << OpticksPhoton::Flag(b_flag)   
        << std::endl 
        << b.desc()
        << std::endl 
        ;
}



/**
SEvt::rjoinPhotonCheck
------------------------

Would expect all rejoin to have BULK_ABSORB, but with 
very artifical single volume of Scintillator geometry keep getting photons 
hitting world boundary giving MISS. 

What about perhaps reemission immediately after reemission ? Nope, I think 
a new point would always be minted so the current_photon flag should 
never be BULK_REEMIT when rjoin runs. 

**/

void SEvt::rjoinPhotonCheck(const sphoton& ph ) const 
{
    dbg->rjoinPhotonCheck++ ; 

    unsigned flag = ph.flag();
    unsigned flagmask = ph.flagmask ; 

    bool flag_AB = flag == BULK_ABSORB ;  
    bool flag_MI = flag == MISS ;  
    bool flag_xor = flag_AB ^ flag_MI ;  

    if(flag_AB) dbg->rjoinPhotonCheck_flag_AB++ ; 
    if(flag_MI) dbg->rjoinPhotonCheck_flag_MI++ ; 
    if(flag_xor) dbg->rjoinPhotonCheck_flag_xor++ ; 

    bool flagmask_AB = flagmask & BULK_ABSORB  ; 
    bool flagmask_MI = flagmask & MISS ; 
    bool flagmask_or = flagmask_AB | flagmask_MI ; 

    if(flagmask_AB) dbg->rjoinPhotonCheck_flagmask_AB++ ; 
    if(flagmask_MI) dbg->rjoinPhotonCheck_flagmask_MI++ ; 
    if(flagmask_or) dbg->rjoinPhotonCheck_flagmask_or++ ; 
    
    bool expect = flag_xor && flagmask_or ; 
    if(!expect) LOG(fatal)
        << "rjoinPhotonCheck : unexpected flag/flagmask" 
        << " flag_AB " << flag_AB
        << " flag_MI " << flag_MI
        << " flag_xor " << flag_xor
        << " flagmask_AB " << flagmask_AB
        << " flagmask_MI " << flagmask_MI
        << " flagmask_or " << flagmask_or
        << ph.descFlag()
        << std::endl
        << ph.desc()
        << std::endl
        ;
    assert(expect); 

}

void SEvt::rjoinSeqCheck(unsigned seq_flag) const 
{
    dbg->rjoinSeqCheck++ ; 

    bool flag_AB = seq_flag == BULK_ABSORB ;
    bool flag_MI = seq_flag == MISS ;
    bool flag_xor = flag_AB ^ flag_MI ;          

    if(flag_AB)  dbg->rjoinSeqCheck_flag_AB++ ; 
    if(flag_MI)  dbg->rjoinSeqCheck_flag_MI++ ; 
    if(flag_xor) dbg->rjoinSeqCheck_flag_xor++ ; 

    if(!flag_xor) LOG(fatal) << " flag_xor FAIL " << OpticksPhoton::Abbrev(seq_flag) << std::endl ;  
    assert( flag_xor ); 
}

/**
SEvt::pointPhoton
------------------

Invoked from U4Recorder::UserSteppingAction_Optical to cause the 
current photon to be recorded into record vector. 

The pointPhoton and finalPhoton methods need to do the hostside equivalent of 
what CSGOptiX/CSGOptiX7.cu:simulate does device side,
so have setup the environment to match::

ctx.point looks like it is populating sevent arrays, but actually are also populating 
the vectors thanks to setting of the sevent pointers in SEvt::hostside_running_resize
so that the pointers follow around the vectors as they get reallocated. 

TODO: truncation : bounce < max_bounce 
at truncation ctx.point/ctx.trace stop writing anything but bounce keeps incrementing 
**/

void SEvt::pointPhoton(const spho& label)
{
    dbg->pointPhoton++ ; 

    assert( label.isSameLineage(current_pho) ); 
    unsigned idx = label.id ; 
    sctx& ctx = current_ctx ; 
    assert( ctx.idx == idx ); 
    bool first_point = ctx.p.flagmask_count() == 1 ; 
    int& bounce = slot[idx] ; 

    if(first_point == false) ctx.trace(bounce); 
    ctx.point(bounce); 

    sseq& seq = ctx.seq ;  

    LOG(LEVEL) 
        << " idx " << idx 
        << " bounce " << bounce 
        << " first_point " << first_point 
        << " evt.max_record " << evt->max_record
        << " evt.max_rec    " << evt->max_rec
        << " evt.max_seq    " << evt->max_seq
        << " evt.max_prd    " << evt->max_prd
        << " evt.max_tag    " << evt->max_tag
        << " evt.max_flat    " << evt->max_flat
        << " label.desc " << label.desc() 
        << " seq.desc_seqhis " << seq.desc_seqhis() ; 
        ;

    bounce += 1 ; 
}

/**
SEvt::addTag
-------------

HMM: this needs to be called following every random consumption ... see U4Random::flat 

**/

void SEvt::addTag(unsigned tag, float flat)
{
    if(evt->tag == nullptr) return  ; 
    stagr& tagr = current_ctx.tagr ; 
    unsigned idx = current_ctx.idx ; 

    if( int(idx) == PIDX ) 
    {
        LOG(info) 
             << " idx " << idx
             << " PIDX " << PIDX
             << " tag " << tag 
             << " flat " << flat 
             << " evt.tag " << evt->tag 
             << " tagr.slot " << tagr.slot
              ; 
    }
    tagr.add(tag,flat)  ; 


    if(random)
    {
        int flat_cursor = random->getFlatCursor(); 
        assert( flat_cursor > -1 ); 
        double flat_prior = random->getFlatPrior(); 
        bool cursor_slot_match = unsigned(flat_cursor) == tagr.slot ; 
        if(!cursor_slot_match)
        {
            LOG(error)
                << " idx " << idx
                << " cursor_slot_match " << cursor_slot_match
                << " flat " << flat 
                << " tagr.slot " << tagr.slot 
                << " ( from SRandom "
                << " flat_prior " << flat_prior 
                << " flat_cursor " << flat_cursor 
                << "  ) " 
                << std::endl
                << " MISMATCH MEANS ONE OR MORE PRIOR CONSUMPTIONS WERE NOT TAGGED "
                ;
                ; 
        }
        assert( cursor_slot_match ); 
    }



}

int SEvt::getTagSlot() const 
{
    if(evt->tag == nullptr) return -1 ; 
    const stagr& tagr = current_ctx.tagr ; 
    return tagr.slot ; 
}


/**
SEvt::finalPhoton
--------------------

Canonically called from U4Recorder::PostUserTrackingAction_Optical

**/

void SEvt::finalPhoton(const spho& label)
{
    dbg->finalPhoton++ ; 
    LOG(LEVEL) << label.desc() ; 
    assert( label.isSameLineage(current_pho) ); 
    unsigned idx = label.id ; 
    sctx& ctx = current_ctx ; 
    assert( ctx.idx == idx ); 

    ctx.end(); 
    evt->photon[idx] = ctx.p ;
}

void SEvt::checkPhotonLineage(const spho& label) const 
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
NP* SEvt::getPrd() const 
{ 
    if( evt->prd == nullptr ) return nullptr ; 
    NP* p = makePrd(); 
    p->read2( (float*)evt->prd ); 
    return p ; 
} 
NP* SEvt::getTag() const 
{ 
    if( evt->tag == nullptr ) return nullptr ; 
    NP* p = makeTag(); 
    p->read2( (unsigned long long*)evt->tag ); 
    return p ; 
} 

NP* SEvt::getFlat() const 
{ 
    if( evt->flat == nullptr ) return nullptr ; 
    NP* p = makeFlat(); 
    p->read2( (float*)evt->flat ); 
    return p ; 
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
NP* SEvt::makePrd() const 
{
    return NP::Make<float>( evt->num_photon, evt->max_prd, 2, 4); 
}

/**
SEvt::makeTag
---------------

The evt->max_tag are packed into the stag::NSEQ of each *stag* struct 

For example with stag::NSEQ = 2 there are two 64 bit "unsigned long long" 
in the *stag* struct which with 5 bits per tag is room for 2*12 = 24 tags  

**/

NP* SEvt::makeTag() const 
{
    return NP::Make<unsigned long long>( evt->num_photon, stag::NSEQ);   // 
}
NP* SEvt::makeFlat() const 
{
    return NP::Make<float>( evt->num_photon, sflat::SLOTS );   // 
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
        case SCOMP_INPHOTON:  a = getInputPhoton()   ; break ;   
        case SCOMP_PHOTON:    a = getPhoton()   ; break ;   
        case SCOMP_RECORD:    a = getRecord()   ; break ;   
        case SCOMP_REC:       a = getRec()      ; break ;   
        case SCOMP_SEQ:       a = getSeq()      ; break ;   
        case SCOMP_PRD:       a = getPrd()      ; break ;   
        case SCOMP_TAG:       a = getTag()      ; break ;   
        case SCOMP_FLAT:      a = getFlat()     ; break ;   
        //case SCOMP_SEED:      a = getSeed()     ; break ;   
        //case SCOMP_HIT:       a = getHit()      ; break ;   
        //case SCOMP_SIMTRACE:  a = getSimtrace() ; break ;   
        case SCOMP_DOMAIN:    a = getDomain()   ; break ;   
    }   
    return a ; 
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
    const char* savedir = getSaveDir(); 

    std::stringstream ss ; 
    ss << evt->desc()
       << std::endl
       << " savedir " << ( savedir ? savedir : "-" )
       << std::endl
       << dbg->desc()
       ; 
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
        const char* k = SComp::Name(comp);    
        bool comp_skip = (comp & mask) == 0 ; 
        LOG(LEVEL) << " comp " << comp << " k " << k << " comp_skip " << comp_skip ; 
        if(comp_skip) continue ; 
        NP* a = provider->getComponent(comp); 
        LOG(LEVEL) << " a " << ( a ? a->brief() : "-" ) ; 
        if(a == nullptr) continue ;  
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

const char* SEvt::FALLBACK_DIR = "$DefaultOutputDir" ;  // $TMP/ExecutableName/GEOM
const char* SEvt::DefaultDir()
{
    const char* dir_ = SGeo::LastUploadCFBase_OutDir(); 
    const char* dir = dir_ ? dir_ : FALLBACK_DIR  ; 
    return dir ; 
}

void SEvt::save() 
{
    const char* dir = DefaultDir(); 
    LOG(LEVEL) << "DefaultDir " << dir ; 
    save(dir); 
}
void SEvt::save(const char* base, const char* reldir ) 
{
    const char* dir = SPath::Resolve(base, reldir, DIRPATH); 
    save(dir); 
}

void SEvt::save(const char* base, const char* reldir1, const char* reldir2 ) 
{
    const char* dir = SPath::Resolve(base, reldir1, reldir2,  DIRPATH); 
    save(dir); 
}



/**
SEvt::save
------------

If an index has been set with SEvt::setIndex SEvt::SetIndex 
and not unset with SEvt::UnsetIndex SEvt::unsetIndex
then the directory is suffixed with the index::

    /some/directory/n001
    /some/directory/z000
    /some/directory/p001
   
**/

void SEvt::save(const char* dir_) 
{
    const char* dir = index == MISSING_INDEX ? SPath::Resolve(dir_, reldir,  DIRPATH) : SPath::Resolve(dir_, reldir, index, DIRPATH ) ; 
    LOG(LEVEL) << " dir " << dir ; 

    gather_components(); 

    LOG(LEVEL) << descComponent() ; 
    LOG(LEVEL) << descFold() ; 

    fold->save(dir); 

    saveLabels(dir); 
    saveFrame(dir); 
}


/**
SEvt::saveLabels
--------------

**/

void SEvt::saveLabels(const char* dir_) const 
{
    const char* dir = SPath::Resolve(dir_, DIRPATH );  
    LOG(LEVEL) << dir ; 

    NP* a0 = getPho0();  
    LOG(LEVEL) << " a0 " << ( a0 ? a0->sstr() : "-" ) ; 
    if(a0) a0->save(dir, "pho0.npy"); 

    NP* a = getPho();  
    LOG(LEVEL) << " a " << ( a ? a->sstr() : "-" ) ; 
    if(a) a->save(dir, "pho.npy"); 

    NP* g = getGS(); 
    LOG(LEVEL) << " g " << ( g ? g->sstr() : "-" ) ; 
    if(g) g->save(dir, "gs.npy"); 
}


void SEvt::saveFrame(const char* dir_) const 
{
    frame.save(dir_); 
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


