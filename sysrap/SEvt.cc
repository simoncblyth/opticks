#include "PLOG.hh"
#include "NP.hh"
#include "NPFold.h"
#include "SPath.hh"
#include "SGeo.hh"
#include "SEvt.hh"
#include "SEvent.hh"
#include "SEventConfig.hh"
#include "SComp.h"

const plog::Severity SEvt::LEVEL = PLOG::EnvLevel("SEvt", "DEBUG"); 

SEvt* SEvt::INSTANCE = nullptr ; 

SEvt::SEvt()
    :
    fold(new NPFold)
{ 
    INSTANCE = this ; 
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
void SEvt::setCompProvider(const SCompProvider* provider_)
{
    provider = provider_ ; 
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

bool SEvt::RECORD_PHOTON = true ; 

sgs SEvt::addGenstep(const quad6& q)
{
    unsigned offset = getNumPhoton() ; // number of photons in event before this genstep  (actually since last reset) 
    unsigned q_numphoton = q.numphoton() ; 

    sgs s = {} ;                // genstep summary struct 
    s.index = genstep.size() ;  // 0-based genstep index in event (actually since last reset)  
    s.photons = q_numphoton ;   // numphoton in the genstep 
    s.offset = offset ;         // event global photon offset 
    s.gentype = q.gentype() ; 

    current_gs = s ; 

    // gs labels and gensteps in order of collection
    gs.push_back(s) ; 
    genstep.push_back(q) ; 

    if(RECORD_PHOTON)
    {
        unsigned tot_numphoton = offset + q_numphoton ;   // numphotons from all gensteps in event so far plus this one just added
        pho.resize(    tot_numphoton );  
        photon.resize( tot_numphoton ); 
    }

    return s ; 
}

/**
SEvt::beginPhoton
------------------


**/
void SEvt::beginPhoton(const spho& sp)
{
    if(!RECORD_PHOTON) return ; 

    unsigned id = sp.id ; 
    assert( id < pho.size() );  

    pho0.push_back(sp);   // push_back asis
    pho[id] = sp ;        // slotted in 


    current_pho = sp ; 
    current_photon.zero() ; 
    current_photon.set_idx(id); 
}

/**
SEvt::checkPhoton
-------------------

Called from  U4Recorder::UserSteppingAction

**/

void SEvt::checkPhoton(const spho& sp) const 
{
    assert( sp.isIdentical(current_pho) ); 
}

void SEvt::endPhoton(const spho& sp)
{
    assert( sp.isIdentical(current_pho) ); 
    unsigned id = sp.id ; 
    photon[id] = current_photon ; 
}

NP* SEvt::getPho0() const { return NP::Make<int>( (int*)pho0.data(), int(pho0.size()), 4 ); }
NP* SEvt::getPho() const {  return NP::Make<int>( (int*)pho.data(), int(pho.size()), 4 ); }
NP* SEvt::getGS() const {   return NP::Make<int>( (int*)gs.data(),  int(gs.size()), 4 );  }
NP* SEvt::getPhoton() const { return NP::Make<float>( (float*)photon.data(), int(photon.size()), 4, 4 ); } 


/**
SEvt::savePho
--------------

TODO: this just temporary, should be using NPFold for standardized persisting 

**/

void SEvt::savePho(const char* dir_) const 
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

    NP* p = getPhoton(); 
    LOG(info) << " p " << ( p ? p->sstr() : "-" ) ; 
    if(p) p->save(dir, "p.npy"); 

}



/**
SEvt::getGenstep
-----------------

The returned array takes a full copy of the genstep quad6 vector
with all gensteps collected since the last SEvt::clear. 
The array is thus independent from quad6 vector, and hence is untouched
by SEvt::clear 

**/

NP* SEvt::getGenstep() const 
{
    unsigned num_gs = genstep.size() ; 
    NP* a = nullptr ; 
    if(num_gs > 0)
    {
        a = NP::Make<float>( num_gs, 6, 4 );  
        a->read2( (float*)genstep.data() );  
    }
    return a ; 
}

void SEvt::saveGenstep(const char* dir) const  // HMM: NOT THE STANDARD SAVE 
{
    NP* a = getGenstep(); 
    if(a == nullptr) return ; 
    LOG(LEVEL) << a->sstr() << " dir " << dir ; 
    a->save(dir, "gs.npy"); 
}


std::string SEvt::desc() const 
{
    std::stringstream ss ; 
    for(unsigned i=0 ; i < getNumGenstep() ; i++) ss << gs[i].desc() << std::endl ; 
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
const char* SEvt::DefaultDir()   // TODO: DOES NOT BELONG : MOVE TO SEvt 
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


