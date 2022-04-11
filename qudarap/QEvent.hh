#pragma once

struct qevent ; 
struct quad4 ;
struct qat4 ; 
struct quad6 ;
struct NP ; 

//template <typename T> struct Tran ; 
template <typename T> struct QBuf ; 

#include <vector>
#include <string>
#include "plog/Severity.h"
#include "QUDARAP_API_EXPORT.hh"

/**
QEvent
=======

TODO: follow OEvent technique of initial allocation and resizing at each event 

Unlike typical CPU side event classes am thinking of QEvent/qevent being rather "static" 
with long lived buffers of defined maximum capacity that get reused for each launch.

setGensteps currently using QBuf::Upload which allocates every time, so that is GPU leaking  

* see optixrap/OEvent.cc OEvent::resizeBuffers OContext::resizeBuffer
* HMM: CUDA has no realloc the buffer resize is an OptiX < 7 extension 
* https://stackoverflow.com/questions/5632247/allocating-more-memory-to-an-existing-global-memory-array

* suggests should define maximum buffer sizes as calculated from max number of photons for each launch 
  and arrange that lanches to never exceed those maximums 

* hence all GPU buffers can get allocated once at initialization 
  with the configured maximum sizes and simply get reused from event to event 
  (or more specifically from launch to launch which might not map one-to-one with events)

* this simplifies memory handling as free-ing only needed at termination 

* so there is no need for resizing, other than changing a CPU/GPU side constant eg *num_photons* 
* just need to ensure that launches are always arranged to be below the max

* how to decide the maximums ? depends on available VRAM and also should be user configurable
  within some range 





**/

struct QUDARAP_API QEvent
{
    static constexpr const int M = 1000000 ; 
    static constexpr const int K = 1000 ; 
    static void  CheckGensteps(const NP* gs); 
    static const plog::Severity LEVEL ; 
    static const QEvent* INSTANCE ; 
    static const QEvent* Get(); 

    QEvent(int max_genstep_=100*K, int max_photon_=1*M); 
    void init(); 
    void uploadEvt(); 

    int          max_genstep ; 
    int          max_photon ; 
    qevent*      evt ; 
    qevent*      d_evt ; 
    const NP*    gs ;  
    std::string  meta ; 

    //QBuf<float>* genstep ; 
    //QBuf<int>*   seed  ;

    void setGensteps(const NP* gs);
    std::string descGensteps(int edgeitems=5) const ; 
    //void setGensteps(QBuf<float>* dgs ); 

    unsigned count_genstep_photons(); 
    void     fill_seed_buffer(); 


    void setMeta( const char* meta ); 
    bool hasMeta() const ; 

    void downloadPhoton( std::vector<quad4>& photon ); 
    void savePhoton( const char* dir, const char* name); 
    void saveGenstep(const char* dir, const char* name); 
    void saveMeta(   const char* dir, const char* name); 
 
    void checkEvt() ;  // GPU side 

    qevent* getDevicePtr() const ;
    unsigned getNumPhotons() const ;  
    std::string desc() const ; 
};



