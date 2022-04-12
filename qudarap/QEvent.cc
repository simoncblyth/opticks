#include <cuda_runtime.h>
#include <sstream>

#include "SPath.hh"
#include "scuda.h"
#include "squad.h"
#include "sqat4.h"
#include "stran.h"

#include "SEvent.hh"
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

std::string QEvent::DescGensteps(const NP* gs, int edgeitems) // static 
{
    int num_genstep = gs ? gs->shape[0] : 0 ; 

    quad6* gs_v = (quad6*)gs->cvalues<float>() ; 
    std::stringstream ss ; 
    ss << "QEvent::DescGensteps gs.shape[0] " << num_genstep << " (" ; 

    int total = 0 ; 
    for(int i=0 ; i < num_genstep ; i++)
    {
        const quad6& _gs = gs_v[i]; 
        unsigned gs_pho = _gs.q0.u.w  ; 

        if( i < edgeitems || i > num_genstep - edgeitems ) ss << gs_pho << " " ; 
        else if( i == edgeitems )  ss << "... " ; 

        total += gs_pho ; 
    } 
    ss << ") total " << total  ; 
    std::string s = ss.str(); 
    return s ; 
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
    evt(new qevent),
    d_evt(QU::device_alloc<qevent>(1)),
    gs(nullptr),
    meta()
{
    INSTANCE = this ; 
    init(); 
}

void QEvent::init()
{
    evt->max_genstep = SEventConfig::MaxGenstep() ; 
    evt->max_photon  = SEventConfig::MaxPhoton()  ; 
    evt->max_bounce  = SEventConfig::MaxBounce()  ; 
    evt->max_record  = SEventConfig::MaxRecord()  ; 

    evt->num_genstep = 0 ; 
    evt->num_seed  = 0 ; 
    evt->num_photon = 0 ; 
    evt->num_record = 0 ; 

    evt->genstep = evt->max_genstep > 0 ? QU::device_alloc<quad6>( evt->max_genstep ) : nullptr ; 
    evt->seed    = evt->max_photon  > 0 ? QU::device_alloc<int>(   evt->max_photon )  : nullptr ;   // 1:1 seed:photon
    evt->photon  = evt->max_photon  > 0 ? QU::device_alloc<quad4>( evt->max_photon )  : nullptr ; 
    evt->record  = evt->max_record  > 0 ? QU::device_alloc<quad4>( evt->max_photon * evt->max_record  ) : nullptr ; 
}

std::string QEvent::descMax() const
{
    int w = 5 ; 
    std::stringstream ss ; 
    ss 
        << "QEvent::descMax " 
        << " evt.max_genstep " << std::setw(w) << evt->max_genstep  
        << " evt.max_photon  " << std::setw(w) << evt->max_photon  
        << " evt.max_bounce  " << std::setw(w) << evt->max_bounce 
        << " evt.max_record  " << std::setw(w) << evt->max_record 
        ;
    return ss.str(); 
}

std::string QEvent::desc() const
{
    int w = 5 ; 
    std::stringstream ss ; 
    ss 
        << " QEvent " 
        << " evt.num_genstep " << std::setw(w) << evt->num_genstep 
        << " evt.num_seed "    << std::setw(w) << evt->num_seed   
        << " evt.num_photon "  << std::setw(w) << evt->num_photon
        ;
    return ss.str(); 
}

void QEvent::setMeta(const char* meta_)
{
    meta = meta_ ; 
} 

bool QEvent::hasMeta() const 
{
    return meta.empty() == false ; 
}

void QEvent::CheckGensteps(const NP* gs)  // static
{ 
    assert( gs->uifc == 'f' && gs->ebyte == 4 ); 
    assert( gs->has_shape(-1, 6, 4) ); 
}


/**
QEvent::setGensteps
--------------------

1. gensteps uploaded to QEvent::init allocated evt->genstep device buffer, 
   overwriting any prior gensteps and evt->num_genstep is set 

2. *count_genstep_photons* calculates the total number of seeds (and photons) by 
   adding the photons from each genstep and setting evt->num_seed

3. *fill_seed_buffer* populates seed buffer using num photons per genstep from genstep buffer

3. invokes setNumPhoton


* HMM: find that without zeroing the seed buffer the seed filling gets messed up causing QEventTest fails 
  doing this in QEvent::init is not sufficient need to do in QEvent::setGensteps.  
  **This is a documented limitation of sysrap/iexpand.h**
 
  So far it seems that no zeroing is needed for the genstep buffer. 

**/

void QEvent::setGensteps(const NP* gs_) 
{ 
    gs = gs_ ; 
    CheckGensteps(gs); 
    evt->num_genstep = gs->shape[0] ; 
    LOG(LEVEL) << DescGensteps(gs, 10) ;
 
    bool num_gs_allowed = evt->num_genstep <= evt->max_genstep ;
    if(!num_gs_allowed) LOG(fatal) << " evt.num_genstep " << evt->num_genstep << " evt.max_genstep " << evt->max_genstep ; 
    assert( num_gs_allowed ); 

    quad6* gs_h = (quad6*)gs->cvalues<float>() ; 
   
    QU::copy_host_to_device<quad6>( evt->genstep, gs_h, evt->num_genstep ); 

    QU::device_memset<int>(   evt->seed,    0, evt->max_photon );

    //count_genstep_photons();   // sets evt->num_seed
    //fill_seed_buffer() ;       // populates seed buffer
    count_genstep_photons_and_fill_seed_buffer();   // combi-function doing what both the above do 

    setNumPhoton( evt->num_seed ); 
}

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
QEvent::setNumPhoton
---------------------

Canonically invoked internally from QEvent::setGensteps but may be invoked 
directly from "friendly" photon only tests without use of gensteps.  

Sets evt->num_photon asserts that is within allowed *evt->max_photon* and calls *uploadEvt*

**/

void QEvent::setNumPhoton(unsigned num_photon )
{
    evt->num_photon = num_photon  ; 
    bool num_photon_allowed = evt->num_photon <= evt->max_photon ; 
    if(!num_photon_allowed) LOG(fatal) << " evt.num_photon " << evt->num_photon << " evt.max_photon " << evt->max_photon ; 
    assert( num_photon_allowed ); 
    uploadEvt(); 
}

unsigned QEvent::getNumPhoton() const
{
    return evt->num_photon ; 
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
void QEvent::downloadPhoton( std::vector<quad4>& photon )
{
    if( evt->photon == nullptr ) return ; 
    photon.resize(evt->num_photon); 
    QU::copy_device_to_host<quad4>( photon.data(), evt->photon, evt->num_photon ); 
}
void QEvent::downloadRecord( std::vector<quad4>& record )
{
    if( evt->record == nullptr ) return ; 
    record.resize(evt->num_record); 
    QU::copy_device_to_host<quad4>( record.data(), evt->record, evt->num_record ); 
}





void QEvent::savePhoton( const char* dir_, const char* name )
{
    int create_dirs = 2 ;  // 2:dirpath 
    const char* dir = SPath::Resolve(dir_, create_dirs); 

    LOG(info) << dir ; 
    std::vector<quad4> photon ; 
    downloadPhoton(photon); 
    NP::Write( dir, name,  (float*)photon.data(), photon.size(), 4, 4  );
}

void QEvent::saveGenstep( const char* dir_, const char* name)
{
    if(!gs) return ; 
    int create_dirs = 1 ;  // 1:filepath 
    const char* path = SPath::Resolve(dir_, name, create_dirs); 
    gs->save(path); 
}

void QEvent::saveMeta( const char* dir_, const char* name)
{     
    if(!hasMeta()) return ; 
    int create_dirs = 1 ;  // 1:filepath 
    const char* path = SPath::Resolve(dir_, name, create_dirs); 
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

