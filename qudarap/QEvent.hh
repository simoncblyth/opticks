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
    static void  CheckGensteps(const NP* gs); 
    static const plog::Severity LEVEL ; 
    static QEvent* INSTANCE ; 
    static QEvent* Get(); 
    static std::string DescGensteps(const NP* gs, int edgeitems=5) ; 
    static std::string DescSeed( const std::vector<int>& seed, int edgeitems ); 

    QEvent(); 
    void init(); 

    // NB members needed on both CPU+GPU or from the QEvent.cu functions 
    // should reside inside the qevent.h instance not up here in QEvent.hh
  
    qevent*      evt ; 
    qevent*      d_evt ; 
    const NP*    gs ;  
    std::string  meta ; 

    void     setGensteps(const NP* gs);
    unsigned count_genstep_photons(); 
    void     fill_seed_buffer(); 
    void     count_genstep_photons_and_fill_seed_buffer(); 

    void     setNumPhoton(unsigned num_photon) ;  
    void     uploadEvt(); 
    unsigned getNumPhoton() const ;  

    void downloadGenstep( std::vector<quad6>& genstep ); 
    void downloadSeed(    std::vector<int>&   seed ); 
    void downloadPhoton(  std::vector<quad4>& photon ); 
    void downloadRecord(  std::vector<quad4>& record ); 

    void savePhoton( const char* dir, const char* name); 
    void saveGenstep(const char* dir, const char* name); 

    std::string descMax() const ; 
    std::string desc() const ; 

    void saveMeta(   const char* dir, const char* name); 
    void setMeta( const char* meta ); 
    bool hasMeta() const ; 

    void checkEvt() ;  // GPU side 

    qevent* getDevicePtr() const ;
};


