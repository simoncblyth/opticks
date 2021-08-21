#include <cuda_runtime.h>

#include "scuda.h"
#include "squad.h"
#include "QBuf.hh"

#include "PLOG.hh"
#include "QEvent.hh"
#include "QBuf.hh"
#include "QSeed.hh"
#include "QU.hh"

#include "qevent.h"

template struct QBuf<quad6> ; 

const plog::Severity QEvent::LEVEL = PLOG::EnvLevel("QEvent", "INFO"); 
const QEvent* QEvent::INSTANCE = nullptr ; 
const QEvent* QEvent::Get(){ return INSTANCE ; }

QEvent::QEvent()
    :
    evt(new qevent),
    d_evt(QU::device_alloc<qevent>(1)),
    gensteps(nullptr),
    seeds(nullptr),
    num_photons(0)
{
    INSTANCE = this ; 
}
void QEvent::setGenstepsFake(const std::vector<int>& photon_counts_per_genstep )
{
    LOG(LEVEL) << "[" ; 
    QBuf<quad6>* gs = QSeed::UploadFakeGensteps(photon_counts_per_genstep) ;
    setGensteps(gs); 
    LOG(LEVEL) << "]" ; 
}

void QEvent::setGensteps(QBuf<quad6>* gs_ )
{
    gensteps = gs_ ; 
    seeds = QSeed::CreatePhotonSeeds(gensteps);
    num_photons = seeds->num_items ; 
}

std::string QEvent::desc() const
{
    std::stringstream ss ; 
    ss 
        << " gensteps " << ( gensteps ? gensteps->desc() : "-" ) 
        << " seeeds " << ( seeds ? seeds->desc() : "-" ) 
        ;
    return ss.str(); 
}

void QEvent::uploadEvt()  // Evt is tiny, just a few pointers
{ 
    evt->gs = gensteps->ptr ; 
    evt->se = seeds->ptr ; 
    QU::copy_host_to_device<qevent>(d_evt, evt, 1 );  
}

qevent* QEvent::getDevicePtr() const
{
    return d_evt ; 
}
unsigned QEvent::getNumPhotons() const
{
    return num_photons ; 
}


extern "C" void QEvent_checkEvt(dim3 numBlocks, dim3 threadsPerBlock, qevent* evt, unsigned width, unsigned height ) ; 

void QEvent::checkEvt() 
{ 
    unsigned width = seeds->num_items ; 
    unsigned height = 1 ; 
    LOG(info) << " width " << width << " height " << height ; 

    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 
    QU::ConfigureLaunch( numBlocks, threadsPerBlock, width, height ); 
 
    assert( d_evt ); 
    QEvent_checkEvt(numBlocks, threadsPerBlock, d_evt, width, height );   
}




