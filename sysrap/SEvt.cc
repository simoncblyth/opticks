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

SEvt::SEvt(){ INSTANCE = this ; }

SEvt* SEvt::Get(){ return INSTANCE ; }

sgs SEvt::AddGenstep(const quad6& q)
{
    if(INSTANCE == nullptr) std::cout << "FATAL: must instanciate SEvt before SEvt::AddGenstep  " << std::endl ; 
    assert(INSTANCE); 
    return INSTANCE->addGenstep(q); 
}
sgs SEvt::AddGenstep(const NP* a)
{
    if(INSTANCE == nullptr) std::cout << "FATAL: must instanciate SEvt before SEvt::AddGenstep  " << std::endl ; 
    assert(INSTANCE); 
    return INSTANCE->addGenstep(a); 
}

void SEvt::Clear()
{
    if(INSTANCE == nullptr) std::cout << "FATAL: must instanciate SEvt before SEvt::Clear  " << std::endl ; 
    assert(INSTANCE); 
    INSTANCE->clear(); 
}

void SEvt::Save()
{
    if(INSTANCE == nullptr) std::cout << "FATAL: must instanciate SEvt before SEvt::Save  " << std::endl ; 
    assert(INSTANCE); 
    INSTANCE->save(); 
}
void SEvt::Save(const char* dir)
{
    if(INSTANCE == nullptr) std::cout << "FATAL: must instanciate SEvt before SEvt::Save  " << std::endl ; 
    assert(INSTANCE); 
    INSTANCE->save(dir); 
}
void SEvt::Save(const char* dir, const char* rel)
{
    if(INSTANCE == nullptr) std::cout << "FATAL: must instanciate SEvt before SEvt::Save  " << std::endl ; 
    assert(INSTANCE); 
    INSTANCE->save(dir, rel ); 
}










void SEvt::AddCarrierGenstep()
{
    AddGenstep(SEvent::MakeCarrierGensteps());
}
void SEvt::AddTorchGenstep()
{
    AddGenstep(SEvent::MakeTorchGensteps());
}






int SEvt::GetNumPhoton()
{
   return INSTANCE ? INSTANCE->getNumPhoton() : -1 ;
}

NP* SEvt::GetGenstep() 
{
   return INSTANCE ? INSTANCE->getGenstep() : nullptr ;
}

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

sgs SEvt::addGenstep(const quad6& q)
{
    sgs s = {} ;   // genstep summary struct 

    s.index = genstep.size() ;  // 0-based genstep index in event (actually since last reset)  
    s.photons = q.numphoton() ;  
    s.offset = getNumPhoton() ; // number of photons in event before this genstep  (actually single last reset)  
    s.gentype = q.gentype() ; 

    gs.push_back(s) ; 
    genstep.push_back(q) ; 

    return s ; 
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
        const char* k = SComp::Name(comp);    
        fold->add(k, a); 
    }
    fold->meta = provider->getMeta();  
    // persisted metadata will now be in NPFold_meta.txt (previously fdmeta.txt)
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


