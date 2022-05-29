#include <cuda_runtime.h>
#include <sstream>

#include "SGeo.hh"
#include "SPath.hh"
#include "scuda.h"
#include "squad.h"
#include "sphoton.h"
#include "srec.h"
#include "sseq.h"
#include "sqat4.h"
#include "stran.h"
#include "SU.hh"

#include "SComp.h"
#include "SGenstep.hh"
#include "SEvent.hh"
#include "SEvt.hh"
#include "SEventConfig.hh"
#include "NP.hh"
#include "PLOG.hh"

#include "OpticksGenstep.h"

#include "QEvent.hh"
#include "QBuf.hh"
#include "QBuf.hh"
#include "QU.hh"

#include "qevent.h"

template struct QBuf<quad6> ; 

const plog::Severity QEvent::LEVEL = PLOG::EnvLevel("QEvent", "DEBUG"); 
QEvent* QEvent::INSTANCE = nullptr ; 
QEvent* QEvent::Get(){ return INSTANCE ; }

const char* QEvent::FALLBACK_DIR = "$TMP" ; 
const char* QEvent::DefaultDir() 
{
    const char* dir_ = SGeo::LastUploadCFBase_OutDir(); 
    const char* dir = dir_ ? dir_ : FALLBACK_DIR  ; 
    return dir ; 
}


std::string QEvent::DescSeed( const std::vector<int>& seed, int edgeitems )  // static 
{
    int num_seed = int(seed.size()) ; 

    std::stringstream ss ; 
    ss << "QEvent::DescSeed seed.size " << num_seed << " (" ;

    for(int i=0 ; i < num_seed ; i++)
    {
        if( i < edgeitems || i > num_seed - edgeitems ) ss << seed[i] << " " ; 
        else if( i == edgeitems )  ss << "... " ; 
    } 
    ss << ")"  ; 
    std::string s = ss.str(); 
    return s ; 
}



/**
QEvent::QEvent
----------------

Instanciation allocates device buffers with sizes configured by SEventConfig

**/


QEvent::QEvent()
    :
    selector(new sphoton_selector(SEventConfig::HitMask())),
    evt(new qevent),
    d_evt(QU::device_alloc<qevent>(1)),
    gs(nullptr),
    meta()
{
    INSTANCE = this ; 
    init(); 
}

/**
QEvent::init
--------------

Only configures limits, no allocation yet. Allocation happens in QEvent::setGenstep QEvent::setNumPhoton

**/

void QEvent::init()
{
    evt->max_genstep = SEventConfig::MaxGenstep() ; 
    evt->max_photon  = SEventConfig::MaxPhoton()  ; 
    evt->max_simtrace  = SEventConfig::MaxSimtrace()  ; 
    evt->max_bounce  = SEventConfig::MaxBounce()  ; 
    evt->max_record  = SEventConfig::MaxRecord()  ;  // full step record
    evt->max_rec     = SEventConfig::MaxRec()  ;     // compressed step record 
    evt->max_seq     = SEventConfig::MaxSeq()  ;     // seqhis 

    evt->zero(); 
    LOG(fatal) << descBuf() ; 

    float extent = SEventConfig::MaxExtent() ; 
    float time_max = SEventConfig::MaxTime() ; 

    evt->init_domain( extent, time_max );  


    LOG(fatal) << descMax() ; 
}


NP* QEvent::getDomain() const 
{
    quad4 dom[2] ; 
    evt->get_domain(dom[0]); 
    evt->get_config(dom[1]); 
    NP* domain = NP::Make<float>( 2, 4, 4 ); 
    domain->read2<float>( (float*)&dom[0] ); 
    // actually makes more sense to be on the domain than on the hits 
    // as domain will always be there 
    domain->set_meta<unsigned>("hitmask", selector->hitmask );  
    domain->set_meta<std::string>("creator", "QEvent::getDomain" );  
    return domain ; 
}


std::string QEvent::desc() const
{
    std::stringstream ss ; 
    ss << descMax() << std::endl ;
    ss << descBuf() << std::endl ;
    ss << descNum() << std::endl ;
    std::string s = ss.str();  
    return s ; 
}

std::string QEvent::descMax() const
{
    // TODO: move imp into qevent
    int w = 5 ; 
    std::stringstream ss ; 
    ss 
        << "QEvent::descMax " 
        << " evt.max_genstep " << std::setw(w) << evt->max_genstep  
        << " evt.max_photon  " << std::setw(w) << evt->max_photon  
        << " evt.max_simtrace  " << std::setw(w) << evt->max_simtrace  
        << " evt.max_bounce  " << std::setw(w) << evt->max_bounce 
        << " evt.max_record  " << std::setw(w) << evt->max_record 
        << " evt.max_rec  "    << std::setw(w) << evt->max_rec
        ;

    std::string s = ss.str();  
    return s ; 
}

std::string QEvent::descNum() const
{
    // TODO: move imp into qevent
    int w = 5 ; 
    std::stringstream ss ; 
    ss 
        << " QEvent::descNum  " 
        << " evt.num_genstep " << std::setw(w) << evt->num_genstep 
        << " evt.num_seed "    << std::setw(w) << evt->num_seed   
        << " evt.num_photon "  << std::setw(w) << evt->num_photon
        << " evt.num_simtrace "  << std::setw(w) << evt->num_simtrace
        << " evt.num_record "  << std::setw(w) << evt->num_record
        ;
    std::string s = ss.str();  
    return s ; 
}

std::string QEvent::descBuf() const
{
    // TODO: move imp into qevent
    int w = 5 ; 
    std::stringstream ss ; 
    ss 
        << " QEvent::descBuf  " 
        << " evt.genstep " << std::setw(w) << ( evt->genstep ? "Y" : "N" )
        << " evt.seed "    << std::setw(w) << ( evt->seed    ? "Y" : "N" )  
        << " evt.photon "  << std::setw(w) << ( evt->photon  ? "Y" : "N" ) 
        << " evt.simtrace "  << std::setw(w) << ( evt->simtrace  ? "Y" : "N" ) 
        << " evt.record "  << std::setw(w) << ( evt->record  ? "Y" : "N" )
        ;
    std::string s = ss.str();  
    return s ; 
}




void QEvent::setMeta(const char* meta_)
{
    meta = meta_ ;   // std::string
} 

bool QEvent::hasMeta() const 
{
    return meta.empty() == false ; 
}


/**
QEvent::setGenstep
--------------------

Canonically invoked from QSim::simulate and QSim::simtrace 
just prior to cx->launch 


1. gensteps uploaded to QEvent::init allocated evt->genstep device buffer, 
   overwriting any prior gensteps and evt->num_genstep is set 

2. *count_genstep_photons* calculates the total number of seeds (and photons) by 
   adding the photons from each genstep and setting evt->num_seed

3. *fill_seed_buffer* populates seed buffer using num photons per genstep from genstep buffer

3. invokes setNumPhoton which may allocate records


* HMM: find that without zeroing the seed buffer the seed filling gets messed up causing QEventTest fails 
  doing this in QEvent::init is not sufficient need to do in QEvent::setGenstep.  
  **This is a documented limitation of sysrap/iexpand.h**
 
  So far it seems that no zeroing is needed for the genstep buffer. 

HMM: what about simtrace ? ce-gensteps are very different to ordinary gs 

**/

int QEvent::setGenstep()
{
    NP* gs = SEvt::GetGenstep(); 
    SEvt::Clear();   // clear the quad6 vector, ready to collect more genstep
    if(gs == nullptr) LOG(fatal) << "Must SEvt::AddGenstep before calling QEvent::setGenstep " ;
    return gs == nullptr ? -1 : setGenstep(gs) ; 
} 

int QEvent::setGenstep(const NP* gs_) 
{ 
    gs = gs_ ; 
    SGenstep::Check(gs); 
    evt->num_genstep = gs->shape[0] ; 

    if( evt->genstep == nullptr && evt->seed == nullptr ) 
    {
        LOG(info) << " device_alloc genstep and seed " ; 
        evt->genstep = QU::device_alloc<quad6>( evt->max_genstep ) ; 
        evt->seed    = QU::device_alloc<int>(   evt->max_photon )  ;
    }

    LOG(LEVEL) << SGenstep::Desc(gs, 10) ;
 
    bool num_gs_allowed = evt->num_genstep <= evt->max_genstep ;
    if(!num_gs_allowed) LOG(fatal) << " evt.num_genstep " << evt->num_genstep << " evt.max_genstep " << evt->max_genstep ; 
    assert( num_gs_allowed ); 

    QU::copy_host_to_device<quad6>( evt->genstep, (quad6*)gs->bytes(), evt->num_genstep ); 

    QU::device_memset<int>(   evt->seed,    0, evt->max_photon );

    //count_genstep_photons();   // sets evt->num_seed
    //fill_seed_buffer() ;       // populates seed buffer
    count_genstep_photons_and_fill_seed_buffer();   // combi-function doing what both the above do 


    int gencode0 = SGenstep::GetGencode(gs, 0); // gencode of first genstep   

    if(OpticksGenstep_::IsFrame(gencode0))
    {
        setNumSimtrace( evt->num_seed ); 
    }
    else
    {
        setNumPhoton( evt->num_seed );  // photon, rec, record may be allocated here depending on SEventConfig
    }
    return 0 ; 
}

int QEvent::setGenstep(const quad6* qgs, unsigned num_gs ) 
{
    NP* gs_ = NP::Make<float>( num_gs, 6, 4 ); 
    gs_->read2( (float*)qgs );   
    return setGenstep( gs_ ); 
}

const NP* QEvent::getGenstep() const 
{
    return gs ; 
}

bool QEvent::hasGenstep() const { return evt->genstep != nullptr ; }
bool QEvent::hasSeed() const {    return evt->seed != nullptr ; }
bool QEvent::hasPhoton() const {  return evt->photon != nullptr ; }
bool QEvent::hasRecord() const { return evt->record != nullptr ; }
bool QEvent::hasRec() const    { return evt->rec != nullptr ; }
bool QEvent::hasSeq() const    { return evt->seq != nullptr ; }
bool QEvent::hasHit() const    { return evt->hit != nullptr ; }
bool QEvent::hasSimtrace() const  { return evt->simtrace != nullptr ; }




/**
QEvent::count_genstep_photons
------------------------------

thrust::reduce using strided iterator summing over GPU side gensteps 

**/

extern "C" unsigned QEvent_count_genstep_photons(qevent* evt) ; 
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

extern "C" void QEvent_fill_seed_buffer(qevent* evt ); 
void QEvent::fill_seed_buffer()
{
    QEvent_fill_seed_buffer( evt ); 
}

extern "C" void QEvent_count_genstep_photons_and_fill_seed_buffer(qevent* evt ); 
void QEvent::count_genstep_photons_and_fill_seed_buffer()
{
    QEvent_count_genstep_photons_and_fill_seed_buffer( evt ); 
}


/**
QEvent::setPhotons
-------------------

This is only used with non-standard input photon running, 
eg the photon mutatating QSimTest use this.  
The normal mode of operation is to start from gensteps using QEvent::setGenstep
and seed and generate photons on device.

**/

void QEvent::setPhoton(const NP* p_)
{
    p = p_ ; 

    int num_photon = p->shape[0] ; 
    
    LOG(info) << "[ " <<  p->sstr() << " num_photon " << num_photon  ; 

    assert( p->has_shape( -1, 4, 4) ); 

    setNumPhoton( num_photon ); 

    QU::copy_host_to_device<sphoton>( evt->photon, (sphoton*)p->bytes(), num_photon ); 

    LOG(info) << "] " <<  p->sstr() << " num_photon " << num_photon  ; 
}



/**
QEvent::getPhoton(NP* p) :  mutating API
-------------------------------------------
**/

void QEvent::getPhoton(NP* p) const 
{
    LOG(fatal) << "[ evt.num_photon " << evt->num_photon << " p.sstr " << p->sstr() << " evt.photon " << evt->photon ; 
    assert( p->has_shape(evt->num_photon, 4, 4) ); 
    QU::copy_device_to_host<sphoton>( (sphoton*)p->bytes(), evt->photon, evt->num_photon ); 
    LOG(fatal) << "] evt.num_photon " << evt->num_photon  ; 
}

NP* QEvent::getPhoton() const 
{
    NP* p = NP::Make<float>( evt->num_photon, 4, 4);
    getPhoton(p); 
    return p ; 
}

void QEvent::getSimtrace(NP* t) const 
{
    LOG(fatal) << "[ evt.num_simtrace " << evt->num_simtrace << " t.sstr " << t->sstr() << " evt.simtrace " << evt->simtrace ; 
    assert( t->has_shape(evt->num_simtrace, 4, 4) ); 
    QU::copy_device_to_host<quad4>( (quad4*)t->bytes(), evt->simtrace, evt->num_simtrace ); 
    LOG(fatal) << "] evt.num_simtrace " << evt->num_simtrace  ; 
}
NP* QEvent::getSimtrace() const 
{
    NP* t = NP::Make<float>( evt->num_simtrace, 4, 4);
    getSimtrace(t); 
    return t ; 
}





void QEvent::getSeq(NP* seq) const 
{
    if(!hasSeq()) return ; 
    LOG(fatal) << "[ evt.num_seq " << evt->num_seq << " seq.sstr " << seq->sstr() << " evt.seq " << evt->seq ; 
    assert( seq->has_shape(evt->num_seq, 2) ); 
    QU::copy_device_to_host<sseq>( (sseq*)seq->bytes(), evt->seq, evt->num_seq ); 
    LOG(fatal) << "] evt.num_seq " << evt->num_seq  ; 
}

NP* QEvent::getSeq() const 
{
    if(!hasSeq()) return nullptr ;  
    NP* seq = NP::Make<unsigned long long>( evt->num_seq, 2);
    getSeq(seq); 
    return seq ; 
}

NP* QEvent::getRecord() const 
{
    if(!hasRecord()) return nullptr ;  

    NP* r = NP::Make<float>( evt->num_photon, evt->max_record, 4, 4);  // stride: sizeof(float)*4*4 = 4*4*4 = 64
    r->set_meta<std::string>("rpos", "4,GL_FLOAT,GL_FALSE,64,0,false" );  // eg used by examples/UseGeometryShader

    LOG(info) << " evt.num_record " << evt->num_record ; 
    QU::copy_device_to_host<sphoton>( (sphoton*)r->bytes(), evt->record, evt->num_record ); 
    return r ; 
}

NP* QEvent::getRec() const 
{
    if(!hasRec()) return nullptr ;  

    NP* r = NP::Make<short>( evt->num_photon, evt->max_rec, 2, 4);   // stride:  sizeof(short)*2*4 = 2*2*4 = 16   
    r->set_meta<std::string>("rpos", "4,GL_SHORT,GL_TRUE,16,0,false" );  // eg used by examples/UseGeometryShader

    LOG(info) 
        << " evt.num_photon " << evt->num_photon 
        << " evt.max_rec " << evt->max_rec 
        << " evt.num_rec " << evt->num_rec  
        << " evt.num_photon*evt.max_rec " << evt->num_photon*evt->max_rec  
        ;

    assert( evt->num_photon*evt->max_rec == evt->num_rec );  

    QU::copy_device_to_host<srec>( (srec*)r->bytes(), evt->rec, evt->num_rec ); 
    return r ; 
}


unsigned QEvent::getNumHit() const 
{
    assert( evt->photon ); 
    assert( evt->num_photon ); 

    evt->num_hit = SU::count_if_sphoton( evt->photon, evt->num_photon, *selector );    

    LOG(info) << " evt.photon " << evt->photon << " evt.num_photon " << evt->num_photon << " evt.num_hit " << evt->num_hit ;  
    return evt->num_hit ; 
}

/**
QEvent::getHit
------------------

1. count *evt.num_hit* passing the photon *selector* 
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

NP* QEvent::getHit() const 
{
    assert( evt->photon ); 

    assert( evt->num_photon ); 

    evt->num_hit = SU::count_if_sphoton( evt->photon, evt->num_photon, *selector );    

    LOG(info) 
         << " evt.photon " << evt->photon 
         << " evt.num_photon " << evt->num_photon 
         << " evt.num_hit " << evt->num_hit
         << " selector.hitmask " << selector->hitmask
         << " SEventConfig::HitMask " << SEventConfig::HitMask()
         << " SEventConfig::HitMaskLabel " << SEventConfig::HitMaskLabel()
         ;  

    NP* hit = evt->num_hit > 0 ? getHit_() : nullptr ; 

    return hit ; 
}

NP* QEvent::getHit_() const 
{
    evt->hit = QU::device_alloc<sphoton>( evt->num_hit ); 

    SU::copy_if_device_to_device_presized_sphoton( evt->hit, evt->photon, evt->num_photon,  *selector );

    NP* hit = NP::Make<float>( evt->num_hit, 4, 4 ); 

    QU::copy_device_to_host<sphoton>( (sphoton*)hit->bytes(), evt->hit, evt->num_hit );

    QU::device_free<sphoton>( evt->hit ); 

    evt->hit = nullptr ; 
    LOG(info) << " hit.sstr " << hit->sstr() ; 

    return hit ; 
}


/**
QEvent::save
--------------

Canonically invoked from QSim::save 

QEvent::save persists NP arrays into the default directory 
or the directory argument provided.

Unlike its predecessor, OpticksEvent, organization of 
the directory structure is left to the caller.   

TODO: a separate struct to yield such directory paths handling things
like event tags. 

**/

void QEvent::save() const 
{
    const char* dir = DefaultDir(); 
    LOG(info) << "DefaultDir " << dir ; 
    save(dir); 
}


void QEvent::save(const char* base, const char* reldir ) const 
{
    const char* dir = SPath::Resolve(base, reldir, DIRPATH); 
    save(dir); 
}


/**
QEvent::save
--------------

TODO: move array management to SEvt/NPFold 

The component arrays downloaded from the device and passed to SEvt/NPFold 
for saving are now configurable via SEventConfig::SetCompMask (using SComp.h) 

NB SEventConfig maximums and the comp mask need to be coordinated. 

TODO: protections 

**/

void QEvent::save(const char* dir_) const 
{
    const char* dir = SPath::Resolve(dir_, DIRPATH); 
    LOG(info) << " dir " << dir ; 

    unsigned mask = SEventConfig::CompMask(); 

    const NP* genstep = SComp::IsGenstep(mask) ? getGenstep() : nullptr ; 
    NP* hit      = SComp::IsHit(mask)      ? getHit()      : nullptr ;
    NP* photon   = SComp::IsPhoton(mask)   ? getPhoton()   : nullptr ; 
    NP* record   = SComp::IsRecord(mask)   ? getRecord()   : nullptr ; 
    NP* rec      = SComp::IsRec(mask)      ? getRec()      : nullptr ;   // compressed record
    NP* seq      = SComp::IsSeq(mask)      ? getSeq()      : nullptr ; ;
    NP* domain   = SComp::IsDomain(mask)   ? getDomain()   : nullptr ;
    NP* simtrace = SComp::IsSimtrace(mask) ? getSimtrace() : nullptr ;
    // TODO: getSeed 

    LOG(info) << descSave(hit,genstep,photon,record,rec,seq,domain, simtrace) ; 

    if(hit)      hit->save(      dir, "hit.npy"); 
    if(genstep)  genstep->save(  dir, "genstep.npy"); 
    if(photon)   photon->save(   dir, "photon.npy"); 
    if(record)   record->save(   dir, "record.npy"); 
    if(rec)      rec->save(      dir, "rec.npy"); 
    if(seq)      seq->save(      dir, "seq.npy"); 
    if(domain)   domain->save(   dir, "domain.npy"); 
    if(simtrace) simtrace->save( dir, "simtrace.npy"); 

    saveMeta(dir, "fdmeta.txt" );
}

std::string QEvent::descSave( 
    const NP* hit, 
    const NP* genstep, 
    const NP* photon, 
    const NP* record, 
    const NP* rec, 
    const NP* seq, 
    const NP* domain,
    const NP* simtrace
    ) const 
{
    std::stringstream ss ; 
    ss << "QEvent::descSave" 
       << std::endl 
       << std::setw(20) << "hit" << " " 
       << std::setw(20) << ( hit ? hit->sstr() : "-" ) 
       << " "
       << std::endl
       << std::setw(20) << "genstep" << " " 
       << std::setw(20) << ( genstep ? genstep->sstr() : "-" ) 
       << " "
       << std::setw(30) << "SEventConfig::MaxGenstep" 
       << std::setw(20) << evt->max_genstep
       << std::endl
       << std::setw(20) << "photon" << " " 
       << std::setw(20) << ( photon ? photon->sstr() : "-" ) 
       << " "
       << std::setw(30) << "SEventConfig::MaxPhoton"
       << std::setw(20) << evt->max_photon
       << std::endl
       << std::setw(20) << "record" << " " 
       << std::setw(20) << ( record ? record->sstr() : "-" ) 
       << " " 
       << std::setw(30) << "SEventConfig::MaxRecord"
       << std::setw(20) << evt->max_record
       << std::endl
       << std::setw(20) << "rec" << " " 
       << std::setw(20) << ( rec ? rec->sstr() : "-" ) 
       << " "
       << std::setw(30) << "SEventConfig::MaxRec"
       << std::setw(20) << evt->max_rec
       << std::endl
       << std::setw(20) << "seq" << " " 
       << std::setw(20) << ( seq ? seq->sstr() : "-" ) 
       << " " 
       << std::setw(30) << "SEventConfig::MaxSeq"
       << std::setw(20) << evt->max_seq
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


/**
QEvent::setNumPhoton
---------------------

Canonically invoked internally from QEvent::setGenstep but may be invoked 
directly from "friendly" photon only tests without use of gensteps.  

Sets evt->num_photon asserts that is within allowed *evt->max_photon* and calls *uploadEvt*

**/

void QEvent::setNumPhoton(unsigned num_photon )
{
    evt->num_photon = num_photon  ; 
    bool num_photon_allowed = evt->num_photon <= evt->max_photon ; 
    if(!num_photon_allowed) LOG(fatal) << " evt.num_photon " << evt->num_photon << " evt.max_photon " << evt->max_photon ; 
    assert( num_photon_allowed ); 

    if( evt->photon == nullptr ) 
    {
        evt->photon = QU::device_alloc<sphoton>( evt->max_photon ) ; 

        evt->num_seq = evt->max_seq > 0 ? evt->num_photon : 0 ; 
        evt->seq     = evt->num_seq > 0 ? QU::device_alloc_zero<sseq>(  evt->num_seq  ) : nullptr ; 
        
        // assumes that the number of photons for subsequent launches does not increase 
        // when collecting records : that is ok during highly controlled debugging 

        evt->num_record = evt->max_record * evt->num_photon ;  
        evt->record     = evt->num_record  > 0 ? QU::device_alloc_zero<sphoton>( evt->num_record  ) : nullptr ; 


        evt->num_rec    = evt->max_rec * evt->num_photon ;  
        evt->rec        = evt->num_rec  > 0 ? QU::device_alloc_zero<srec>(  evt->num_rec  ) : nullptr ; 


        // use SEventConfig code or envvars to config the maxima

        LOG(info) 
            << " device_alloc photon " 
            << " evt.num_photon " << evt->num_photon 
            << " evt.max_photon " << evt->max_photon
            << " evt.num_record " << evt->num_record 
            << " evt.num_rec    " << evt->num_rec 
            ;
    }

    uploadEvt(); 
}

void QEvent::setNumSimtrace(unsigned num_simtrace)
{
    evt->num_simtrace = num_simtrace ; 
    bool num_simtrace_allowed = evt->num_simtrace <= evt->max_simtrace ; 
    if(!num_simtrace_allowed) LOG(fatal) << " evt.num_simtrace " << evt->num_simtrace << " evt.max_simtrace " << evt->max_simtrace ; 
    assert( num_simtrace_allowed ); 

    if( evt->simtrace == nullptr ) 
    {
        evt->simtrace = QU::device_alloc<quad4>( evt->max_simtrace ) ; 
    
        LOG(info) 
            << " device_alloc simtrace " 
            << " evt.num_simtrace " << evt->num_simtrace 
            << " evt.max_simtrace " << evt->max_simtrace
            ;
    }
    uploadEvt(); 
}
 


unsigned QEvent::getNumPhoton() const
{
    return evt->num_photon ; 
}
unsigned QEvent::getNumSimtrace() const
{
    return evt->num_simtrace ; 
}




/**
QEvent::uploadEvt 
--------------------

Copies host side *evt* instance (with updated num_genstep and num_photon) to device side  *d_evt*.  
Note that the evt->genstep and evt->photon pointers are not updated, so the same buffers are reused for each launch. 

**/

void QEvent::uploadEvt()
{
    QU::copy_host_to_device<qevent>(d_evt, evt, 1 );  
}


/**
QEvent::downloadGenstep
------------------------

Are these needed with the NP getters ?
**/

void QEvent::downloadGenstep( std::vector<quad6>& genstep )
{
    if( evt->genstep == nullptr ) return ; 
    genstep.resize(evt->num_photon); 
    QU::copy_device_to_host<quad6>( genstep.data(), evt->genstep, evt->num_genstep ); 
}
void QEvent::downloadSeed( std::vector<int>& seed )
{
    if( evt->seed == nullptr ) return ; 
    seed.resize(evt->num_seed); 
    QU::copy_device_to_host<int>( seed.data(), evt->seed, evt->num_seed ); 
}



void QEvent::saveGenstep( const char* dir_, const char* name)
{
    if(!gs) return ; 
    int create_dirs = 1 ;  // 1:filepath 
    const char* path = SPath::Resolve(dir_, name, create_dirs); 
    gs->save(path); 
}

void QEvent::saveMeta( const char* dir_, const char* name) const 
{     
    if(!hasMeta()) return ; 
    const char* path = SPath::Resolve(dir_, name, FILEPATH); 
    NP::WriteString(path, meta.c_str() ); 
}

qevent* QEvent::getDevicePtr() const
{
    return d_evt ; 
}


extern "C" void QEvent_checkEvt(dim3 numBlocks, dim3 threadsPerBlock, qevent* evt, unsigned width, unsigned height ) ; 

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




