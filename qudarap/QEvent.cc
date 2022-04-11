#include <cuda_runtime.h>
#include <sstream>

#include "SPath.hh"
#include "scuda.h"
#include "squad.h"
#include "sqat4.h"
#include "stran.h"

#include "SEvent.hh"
#include "NP.hh"
#include "PLOG.hh"

#include "OpticksGenstep.h"

#include "QEvent.hh"
#include "QBuf.hh"
#include "QBuf.hh"
#include "QSeed.hh"
#include "QU.hh"

#include "qevent.h"

template struct QBuf<quad6> ; 

const plog::Severity QEvent::LEVEL = PLOG::EnvLevel("QEvent", "DEBUG"); 
const QEvent* QEvent::INSTANCE = nullptr ; 
const QEvent* QEvent::Get(){ return INSTANCE ; }

/**
QEvent::QEvent
----------------

Canonical instance instanciated by CSGOptiX::CSGOptiX

**/


QEvent::QEvent(int max_genstep_, int max_photon_)
    :
    max_genstep(max_genstep_),
    max_photon(max_photon_),
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
    evt->genstep = QU::device_alloc<quad6>( max_genstep ); 
    evt->seed    = QU::device_alloc<int>(   max_photon ); 
    evt->photon  = QU::device_alloc<quad4>( max_photon ); 

    evt->num_genstep = max_genstep ; 
    evt->num_seed    = max_photon ; 
    evt->num_photon  = max_photon ; 
}


std::string QEvent::desc() const
{
    std::stringstream ss ; 
    ss 
        << " QEvent " 
        << " evt.num_genstep " << ( evt ? evt->num_genstep : -1 ) 
        << " evt.num_seed "    << ( evt ? evt->num_seed : -1 ) 
        << " evt.num_photon "  << ( evt ? evt->num_photon : -1 ) 
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

std::string QEvent::descGensteps(int edgeitems) const 
{
    quad6* gs_v = (quad6*)gs->cvalues<float>() ; 
    std::stringstream ss ; 
    ss << "QEvent::descGensteps evt.num_genstep " << evt->num_genstep << " (" ; 

    int total = 0 ; 
    for(int i=0 ; i < evt->num_genstep ; i++)
    {
        const quad6& _gs = gs_v[i]; 
        unsigned gs_pho = _gs.q0.u.w  ; 

        if( i < edgeitems || i > evt->num_genstep - edgeitems ) ss << gs_pho << " " ; 
        else if( i == edgeitems )  ss << "... " ; 

        total += gs_pho ; 
    } 
    ss << ") total " << total  ; 
    std::string s = ss.str(); 
    return s ; 
}


/**
QEvent::setGensteps
--------------------

**/

void QEvent::setGensteps(const NP* gs_) 
{ 
    gs = gs_ ; 
    CheckGensteps(gs); 
    evt->num_genstep = gs->shape[0] ; 

    LOG(info) << descGensteps() ; 

    bool num_gs_allowed = evt->num_genstep <= max_genstep ;
    if(!num_gs_allowed) LOG(fatal) << " evt.num_genstep " << evt->num_genstep << " max_genstep " << max_genstep ; 
    assert( num_gs_allowed ); 

    quad6* gs_v = (quad6*)gs->cvalues<float>() ; 

    QU::copy_host_to_device<quad6>( evt->genstep, gs_v, evt->num_genstep ); 



    count_genstep_photons();  

    bool num_seed_allowed = evt->num_seed <= max_photon ; 
    if(!num_seed_allowed) LOG(fatal) << " evt.num_seed " << evt->num_seed << " max_photon " << max_photon ; 
    assert( num_seed_allowed ); 

    evt->num_photon = evt->num_seed ; 

    uploadEvt(); 

 //    QBuf<float>* dgs = QBuf<float>::Upload( gs );   
//    setGensteps(dgs); 
}

void QEvent::uploadEvt()
{
    QU::copy_host_to_device<qevent>(d_evt, evt, 1 );  
}


/**
QEvent::setGensteps
---------------------

Currently this is allocating seed array on every call. 

void QEvent::setGensteps(QBuf<float>* genstep_ ) // QBuf::ptr references already uploaded gensteps
{
    genstep = genstep_ ;   
    // HMM: remove excess duplication 
    // evt->genstep = (quad6*)genstep->d ;   NOPE 
    seed = QSeed::CreatePhotonSeeds(genstep);
    if(!seed) LOG(fatal) << " FAILED to QSeed::CreatePhotonSeeds : problem with gensteps ? " ; 
    assert( seed ); 
    evt->seed = seed->d ; 
    evt->num_seed = seed->num_items ; 
    evt->num_photon = seed->num_items ; 
    evt->photon = QU::device_alloc<quad4>(evt->num_photon) ; 
}
**/



void QEvent::downloadPhoton( std::vector<quad4>& photon )
{
    photon.resize(evt->num_photon); 
    QU::copy_device_to_host_and_free<quad4>( photon.data(), evt->photon, evt->num_photon ); 
}


extern "C" unsigned QEvent_count_genstep_photons(qevent* evt) ; 
unsigned QEvent::count_genstep_photons()
{
   return QEvent_count_genstep_photons( evt );  
}

extern "C" void QEvent_fill_seed_buffer(qevent* evt ); 
void QEvent::fill_seed_buffer()
{
    QEvent_fill_seed_buffer( evt ); 
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

unsigned QEvent::getNumPhotons() const
{
    return evt->num_photon ; 
}

extern "C" void QEvent_checkEvt(dim3 numBlocks, dim3 threadsPerBlock, qevent* evt, unsigned width, unsigned height ) ; 

void QEvent::checkEvt() 
{ 
    unsigned width = getNumPhotons() ; 
    unsigned height = 1 ; 
    LOG(info) << " width " << width << " height " << height ; 

    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 
    QU::ConfigureLaunch( numBlocks, threadsPerBlock, width, height ); 
 
    assert( d_evt ); 
    QEvent_checkEvt(numBlocks, threadsPerBlock, d_evt, width, height );   
}

